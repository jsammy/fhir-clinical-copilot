"""Guardrails that run in code, not in the prompt.

Four checks, each answering a different failure:

1. `sanitize_text` — prompt injection via record content. Clinical notes and free-text fields are
   authored by people; anything that reads like an instruction *to the model* is neutralised
   before it reaches a prompt. The patterns are narrow on purpose: a filter that deletes clinical
   text ("will act as health care proxy") does more harm than the attack it looks for.
2. `fence` — the prompt marks record content with <record> tags; record text must not be able to
   close that fence, so angle brackets are neutralised.
3. `check_citations` — fabrication. Every [Resource/id] the model cites must be a resource the
   tools actually returned for *this* patient; uncited sentences in the narrative are rejected.
4. `TOOL_ALLOW_LIST` — capability. The agent calls only these MCP tools; if the server ever
   advertises a tool outside the list (say a write tool added later), the agent refuses to start.

These are deliberately simple and conservative. A regex injection filter is a tripwire, not a
proof: look-alike letters from other alphabets still get through. The structural defences (no data
tools for the model, fixed tool plan, code-written safety sections) are what bound the harm.
"""

from __future__ import annotations

import re
import unicodedata

TOOL_ALLOW_LIST = frozenset(
    {"find_patient", "get_conditions", "get_medications", "get_allergies", "get_observations",
     "get_encounters", "get_notes", "search_notes"}
)
# What the Q&A agent's model may call. find_patient is not here: the patient is resolved by code
# before the model runs, so the model can never choose which patient to look at.
AGENT_TOOLS = TOOL_ALLOW_LIST - {"find_patient"}

# Phrasing aimed at a model, not at a clinician. \s+ so line breaks and double spaces don't evade.
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions|prompts?|rules)",
    r"disregard\s+(the\s+|all\s+|any\s+)?(system|previous|prior|above)",
    r"you\s+are\s+now\b",
    r"\bsystem\s+prompt\b",
    r"\b(act|behave)\s+as\s+(an?\s+|the\s+)?(ai|assistant|language\s+model|chatbot|system|admin(istrator)?)\b",
    r"<\s*/?\s*(system|assistant|user|instructions?|record)\s*>",
    r"\b(call|use|invoke|run)\s+the\s+\w+\s+tool\b",
    r"\b(delete|erase|overwrite|wipe)\s+(the\s+|this\s+|all\s+)?(record|records|chart|patient\s+data)\b",
]
# Deliberately absent: "tell the clinician/doctor ..." and "update the record". Both occur in real
# notes ("asked to tell the doctor if pain worsens"); attacks phrased that way are contained by
# structure (the model never writes the safety lists), not by this filter.
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
REDACTION = "[removed: instruction-like text]"

REF_RE = re.compile(r"^[A-Z][A-Za-z]+/[A-Za-z0-9\-.]{1,64}$")
_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")
# Bracket contents that are trying to be a citation: a "Type/id" or a bare id list like "101, 202".
# Units such as "mm[Hg]" contain neither and are left alone.
_CITATION_LIKE_RE = re.compile(r"/|^\s*\d+(\s*[,;]\s*\d+)*\s*$")
# An explicit refusal is a correct answer with nothing to cite; anything else must cite.
REFUSAL_RE = re.compile(r"^\s*not in (the )?record\.?\s*$", re.IGNORECASE)
# Sentence boundary: ., ! or ? then whitespace then a capital letter.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
# A piece ending in one of these is an abbreviation, not the end of a sentence.
_ABBREVIATIONS = ("Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "St.", "e.g.", "i.e.", "vs.", "approx.", "No.")
# Ids go into query strings: anything outside this set could smuggle extra search parameters.
# At least one letter or digit, so "." and ".." are rejected.
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9.\-]*[A-Za-z0-9][A-Za-z0-9.\-]*$")
MAX_ID_LEN = 64


def sanitize_text(text: str) -> str:
    """Normalise, strip invisible characters, and neutralise instruction-like spans in record text."""
    text = unicodedata.normalize("NFKC", text or "")
    # Cf = Unicode "format" characters: zero-width spaces, joiners, bidi overrides. Invisible to a
    # human reviewer, still read by the model, so they can hide text. Removed before matching.
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = _CONTROL_RE.sub("", text)
    return _INJECTION_RE.sub(REDACTION, text)


def fence(text: str) -> str:
    """Make record text safe to place between <record> tags: no angle brackets, so nothing in the
    record can close the fence or open a fake system/user turn."""
    return text.replace("<", "‹").replace(">", "›")


def check_id(value: str) -> str:
    if not isinstance(value, str) or len(value) > MAX_ID_LEN or not SAFE_ID_RE.match(value):
        raise ValueError(f"rejected id {str(value)[:40]!r}: letters, digits, '-' and '.' only, at least one letter or digit")
    return value


def injection_hits(text: str) -> list[str]:
    return [m.group(0) for m in _INJECTION_RE.finditer(text or "")]


def parse_citations(sentence: str) -> tuple[list[str], list[str]]:
    """(refs, malformed). "[A/1, B/2]" yields both refs, so a fabricated id cannot hide in a list."""
    refs, malformed = [], []
    for content in _BRACKET_RE.findall(sentence):
        if not _CITATION_LIKE_RE.search(content):
            continue
        parts = [p.strip() for p in re.split(r"[,;]", content) if p.strip()]
        if parts and all(REF_RE.match(p) for p in parts):
            refs += parts
        else:
            malformed.append(content)
    return refs, malformed


def split_sentences(items: list[str]) -> list[str]:
    """Models pack several sentences into one list item; each one must carry its own citation.
    Pieces ending in an abbreviation ("Dr.") are joined back to the text that follows."""
    sentences: list[str] = []
    for item in items:
        carry = ""
        for piece in _SENTENCE_END_RE.split(item.strip()):
            piece = f"{carry} {piece}".strip() if carry else piece
            if piece.endswith(_ABBREVIATIONS):
                carry = piece
                continue
            carry = ""
            if piece.strip():
                sentences.append(piece)
        if carry:
            sentences.append(carry)
    return sentences


def check_citations(narrative: list[str], allowed_refs: set[str]) -> list[str]:
    """Return problems found; an empty list means every sentence is grounded in a fetched resource."""
    problems = []
    for i, sentence in enumerate(split_sentences(narrative), 1):
        if REFUSAL_RE.match(sentence):
            continue
        cited, malformed = parse_citations(sentence)
        for bad in malformed:
            problems.append(f"sentence {i} has a malformed citation [{bad}]: use [Type/id], one per bracket")
        if not cited and not malformed:
            problems.append(f"sentence {i} has no citation: {sentence[:80]!r}")
        for ref in cited:
            if ref not in allowed_refs:
                problems.append(f"sentence {i} cites {ref}, which was not retrieved for this patient")
    return problems
