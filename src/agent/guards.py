"""Guardrails that run in code, not in the prompt.

Three checks, each answering a different failure:

1. `sanitize_text` — prompt injection via record content. Clinical notes and free-text fields are
   authored by people; anything that reads like an instruction to the model is neutralised before
   it reaches a prompt, and the prompt itself fences record content as data.
2. `check_citations` — fabrication. Every [Resource/id] the model cites must be a resource the
   tools actually returned for *this* patient; uncited sentences in the narrative are rejected.
3. `TOOL_ALLOW_LIST` — capability. The agent calls only these MCP tools; if the server ever
   advertises a tool outside the list (say a write tool added later), the agent refuses to start.

These are deliberately simple and conservative. A regex injection filter is a tripwire, not a
proof — the structural defences (no write tools, fixed tool plan, deterministic safety sections)
are what bound the harm if a novel injection slips through.
"""

from __future__ import annotations

import re

TOOL_ALLOW_LIST = frozenset(
    {"find_patient", "get_conditions", "get_medications", "get_allergies", "get_observations",
     "get_encounters", "get_notes"}
)

_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (the )?(system|previous|above)",
    r"you are now\b",
    r"\bsystem prompt\b",
    r"\bact as\b",
    r"</?(system|assistant|user|instructions?)>",
    r"\b(call|use|invoke) the \w+ tool\b",
    r"\b(delete|update|modify|write to) (the )?(record|patient|resource)",
]
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
# Sentence boundary: ., ! or ? (optionally after a closing citation bracket) then space + capital.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
# Ids go into query strings: anything outside this set could smuggle extra search parameters.
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9\-.]{1,64}$")


def sanitize_text(text: str) -> str:
    """Strip control characters and neutralise instruction-like spans in record text."""
    text = _CONTROL_RE.sub("", text or "")
    return _INJECTION_RE.sub(REDACTION, text)


def check_id(value: str) -> str:
    if not SAFE_ID_RE.match(value or ""):
        raise ValueError(f"rejected id {value[:40]!r}: only letters, digits, '-' and '.' are allowed")
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
    """Models pack several sentences into one list item; each one must carry its own citation."""
    return [s for item in items for s in _SENTENCE_END_RE.split(item.strip()) if s.strip()]


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
