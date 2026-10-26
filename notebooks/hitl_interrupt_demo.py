"""How interrupt() and resume really behave in LangGraph. Four scenarios, printed step by step.

    python notebooks/hitl_interrupt_demo.py

1. Normal pause and approve: the waiting node re-runs from its first line; interrupt() returns the answer.
2. Restart: a brand-new graph object over the same checkpoint store resumes the paused run.
3. The bug: raising on an invalid decision. The bad answer stays stored at interrupt #0 and is
   replayed on every resume; the corrected answer goes to #1, which the node never reaches.
4. The fix: interrupt() in a loop. The bad answer is consumed, a new interrupt asks again.

Things to try: put a print() or a counter before interrupt() and watch it run twice; resume with a
different thread_id; resume with Command(resume=None) (crashes inside LangGraph 1.0.8).
"""
import asyncio
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class S(TypedDict, total=False):
    draft: str
    decision: dict


def build(review_style: str, saver):
    calls = {"draft": 0, "review": 0}

    async def draft(state: S) -> dict:
        calls["draft"] += 1
        print(f"    [draft] runs (time #{calls['draft']})")
        return {"draft": "Summary text"}

    async def review_raise(state: S) -> dict:
        calls["review"] += 1
        print(f"    [review] node starts from line 1 (time #{calls['review']})")
        decision = interrupt({"show": state["draft"]})
        print(f"    [review] interrupt() RETURNED {decision!r}")
        if not isinstance(decision, dict) or decision.get("action") not in ("approve", "reject"):
            print("    [review] invalid -> raise ValueError")
            raise ValueError("bad decision")
        return {"decision": decision}

    async def review_loop(state: S) -> dict:
        calls["review"] += 1
        print(f"    [review] node starts from line 1 (time #{calls['review']})")
        payload, n = {"show": state["draft"], "error": None}, 0
        while True:
            decision = interrupt(payload)
            print(f"    [review] interrupt() #{n} RETURNED {decision!r}")
            if isinstance(decision, dict) and decision.get("action") in ("approve", "reject"):
                return {"decision": decision}
            payload, n = {**payload, "error": "please answer approve or reject"}, n + 1

    async def finalize(state: S) -> dict:
        print(f"    [finalize] audit written with {state['decision']}")
        return {}

    g = StateGraph(S)
    g.add_node("draft", draft)
    g.add_node("review", review_raise if review_style == "raise" else review_loop)
    g.add_node("finalize", finalize)
    g.add_edge(START, "draft")
    g.add_edge("draft", "review")
    g.add_edge("review", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=saver)


async def show_saved(graph, cfg, label):
    snap = await graph.aget_state(cfg)
    interrupts = [i.value for t in snap.tasks for i in t.interrupts]
    print(f"    saved checkpoint: next={snap.next} values={dict(snap.values)} pending interrupt={interrupts}")


async def step(graph, cfg, label, payload):
    print(f"\n  >>> {label}")
    try:
        result = await graph.ainvoke(payload, cfg)
        if "__interrupt__" in result:
            print(f"    ainvoke RETURNED: paused, payload={result['__interrupt__'][0].value}")
        else:
            print(f"    ainvoke RETURNED: finished, decision={result.get('decision')}")
    except Exception as e:
        print(f"    ainvoke RAISED {type(e).__name__}: {e}")
    await show_saved(graph, cfg, label)


async def main():
    cfg = {"configurable": {"thread_id": "t"}}

    print("\n=== 1. Normal pause and approve ===")
    g = build("raise", InMemorySaver())
    await step(g, cfg, "first call: ainvoke(input)", {"draft": ""})
    await step(g, cfg, "clinician approves: ainvoke(Command(resume=approve))", Command(resume={"action": "approve"}))

    print("\n=== 2. Restart: a brand-new graph object, same checkpoint store ===")
    saver = InMemorySaver()
    g1 = build("raise", saver)
    await step(g1, cfg, "process A: ainvoke(input), then process A exits", {"draft": ""})
    g2 = build("raise", saver)
    await step(g2, cfg, "process B (hours later): Command(resume=approve)", Command(resume={"action": "approve"}))

    print("\n=== 3. The bug: raise on an invalid decision ===")
    g = build("raise", InMemorySaver())
    await step(g, cfg, "first call", {"draft": ""})
    await step(g, cfg, "typo: Command(resume='aprove')", Command(resume="aprove"))
    await step(g, cfg, "corrected: Command(resume=approve)", Command(resume={"action": "approve"}))
    await step(g, cfg, "try again: Command(resume=approve)", Command(resume={"action": "approve"}))

    print("\n=== 4. The fix: interrupt() in a loop ===")
    g = build("loop", InMemorySaver())
    await step(g, cfg, "first call", {"draft": ""})
    await step(g, cfg, "typo: Command(resume='aprove')", Command(resume="aprove"))
    await step(g, cfg, "corrected: Command(resume=approve)", Command(resume={"action": "approve"}))


asyncio.run(main())
