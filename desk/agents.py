"""Analyst and Strategist agents (Claude Sonnet, tool use).

The agents never calculate: every number they state must come from an engine tool result.
`faithfulness()` checks that automatically. Each run is saved to runs/ so the app can fall back
to the last good run if a live call fails or the spend cap is reached.

    python -m desk.agents            # run a live cycle (needs ANTHROPIC_API_KEY), saves runs/latest.json
    DESK_MOCK=1 python -m desk.agents  # offline: deterministic stand-in agents (tests, no key)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from pathlib import Path

from .engine import POLICY, Engine

MODEL = os.environ.get("DESK_MODEL", "claude-sonnet-5-5")
RUNS = Path(__file__).resolve().parents[1] / "runs"
TOP_N = 3                                   # owner call: top 3 opportunities a week

ANALYST_SYSTEM = f"""You are the Analyst agent in a Customer Opportunity Desk for Al Waha Retail, a fictional GCC multi-brand
retail group with six stores named by what they sell: Fashion, Home, Sports, Beauty, Kids, Electronics.
Your user is the customer growth manager. Your job: find the {TOP_N} best CROSS-BRAND opportunities this week:
customers of one store who are likely to start shopping at another store in the group.

Rules:
- Use the tools. Never calculate or estimate numbers yourself; every number you state must appear in a tool result.
- Rank by a combination of affinity strength and audience size; explain the trade-off when they disagree.
- Be honest about weak evidence. Say what you are unsure of.
- Write for a busy manager: plain language, short sentences.
Return ONLY a JSON object: {{"opportunities": [{{"rank": 1, "from_brand": "...", "to_brand": "...",
"headline": "one sentence", "evidence": ["...", "..."], "why_it_matters": "...", "risks": ["..."],
"confidence": "high|medium|low"}}]}}"""

STRATEGIST_SYSTEM = f"""You are the Strategist agent in a Customer Opportunity Desk for Al Waha Retail (fictional, GCC).
Given one cross-brand opportunity, design ONE action for the customer growth manager to approve.

Rules:
- Use the tools. Never calculate numbers yourself; every number you state must appear in a tool result.
- Never add, subtract, divide or compare numbers yourself. To compare options, call compare_options and quote its
  gap_vs_best_aed, share_of_best and discount_premium_over_best_no_discount_aed fields.
- Message first: ALWAYS include no-discount options on at least two channels in compare_options.
  Propose a discount (max {POLICY['max_discount']:.0%}) ONLY if compare_options shows it beats the best no-discount option
  (discount_premium_over_best_no_discount_aed > 0). The north star is incremental gross margin, not revenue.
- The action must pass check_policy. Prefer an action whose control group is measurable; if none is, say it will be
  read as directional and counted in the programme-level result.
- Draft the customer message in English AND Modern Standard Arabic. Warm, specific to the two stores (say e.g. "our Beauty store"), no false urgency,
  no invented facts about products or prices beyond the offer. You may use {{first_name}}. Max 2 short sentences each.
Return ONLY a JSON object: {{"from_brand": "...", "to_brand": "...", "channel": "email|sms|app_push|whatsapp",
"discount": 0.0, "expected_incremental_gross_margin_aed": 0, "eligible_audience": 0, "control_group_share": 0.0,
"measurable": true, "rationale": "2-3 sentences", "alternatives": [{{"channel": "...", "discount": 0.0,
"expected_incremental_gross_margin_aed": 0, "gap_vs_best_aed": 0, "why_not": "..."}}], "message_en": "...", "message_ar": "..."}}"""


def tool_specs(names: list[str]) -> list[dict]:
    brand = {"type": "string", "enum": ["Fashion", "Home", "Sports", "Beauty", "Kids", "Electronics"]}
    pair = {"from_brand": brand, "to_brand": brand}
    action = {**pair, "channel": {"type": "string", "enum": POLICY["channels"]},
              "discount": {"type": "number", "description": "0 to 0.15, e.g. 0.10 for 10% off"}}
    specs = {
        "brand_pair_stats": {"description": "Brand pairs ranked by directional affinity lift (how much more likely "
                             "customers of from_brand are to also shop to_brand than others are), with the audience size "
                             "of from_brand customers active in 90 days who have never bought to_brand.",
                             "input_schema": {"type": "object", "properties": {"top": {"type": "integer"}}}},
        "pair_detail": {"description": "Evidence for one pair: audience, natural 90-day cross-purchase rate, "
                        "to_brand basket and margin, language and city mix, consent rates by channel.",
                        "input_schema": {"type": "object", "properties": pair, "required": list(pair)}},
        "check_policy": {"description": "Applies consent, contact-frequency, discount-ceiling, budget and global-holdout "
                         "rules for an action; returns eligible audience, exclusions, violations, and the planning "
                         "estimate of incremental gross margin and the control-group size for the eligible audience.",
                         "input_schema": {"type": "object", "properties": action, "required": list(action)}},
        "compare_options": {"description": "Evaluates several (channel, discount) options for one pair side by side: policy "
                            "result, eligible audience, expected incremental gross margin, discount cost, control group, and "
                            "each option's gap to the best option. Use this instead of comparing numbers yourself.",
                            "input_schema": {"type": "object", "properties": {**pair, "options": {"type": "array", "items": {
                                "type": "object", "properties": {"channel": action["channel"], "discount": action["discount"]},
                                "required": ["channel", "discount"]}}}, "required": [*pair, "options"]}},
    }
    return [{"name": n, **specs[n]} for n in names]


def call_tool(engine: Engine, name: str, args: dict):
    if name == "brand_pair_stats":
        return engine.brand_pair_stats(top=int(args.get("top", 10)))
    if name == "pair_detail":
        return engine.pair_detail(args["from_brand"], args["to_brand"])
    if name == "compare_options":
        return engine.compare_options(args["from_brand"], args["to_brand"], args["options"])
    if name == "check_policy":
        return engine.check_policy(args["from_brand"], args["to_brand"], args["channel"], float(args["discount"]))
    raise ValueError(f"unknown tool {name}")


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("agent returned no JSON")
    return json.loads(m.group(0))


def run_agent(engine: Engine, system: str, user: str, tools: list[str], client=None, max_turns: int = 12,
              progress: bool = False) -> dict:
    """Tool-use loop. Returns {"output": parsed JSON, "trace": [...], "usage": {...}}."""
    import anthropic
    client = client or anthropic.Anthropic()
    messages = [{"role": "user", "content": user}]
    trace, tin, tout = [], 0, 0
    for _ in range(max_turns):
        r = client.messages.create(model=MODEL, system=system, max_tokens=2500,
                                   tools=tool_specs(tools), messages=messages)
        tin += r.usage.input_tokens
        tout += r.usage.output_tokens
        messages.append({"role": "assistant", "content": r.content})
        if r.stop_reason != "tool_use":
            text = "".join(b.text for b in r.content if b.type == "text")
            return {"output": _parse_json(text), "trace": trace, "usage": {"input_tokens": tin, "output_tokens": tout}}
        results = []
        for b in r.content:
            if b.type == "tool_use":
                if progress:
                    print(f"   · {b.name}({', '.join(f'{k}={v}' for k, v in b.input.items() if k != 'options')})", flush=True)
                out = call_tool(engine, b.name, b.input)
                trace.append({"tool": b.name, "input": b.input, "output": out})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out, default=str)})
        messages.append({"role": "user", "content": results})
    raise RuntimeError("agent did not finish within the turn limit")


# ---------------------------------------------------------------- offline stand-in (tests / no key)
def _mock_cycle(engine: Engine) -> dict:
    opps, actions = [], []
    for i, p in enumerate(engine.brand_pair_stats(top=TOP_N), 1):
        d = engine.pair_detail(p["from_brand"], p["to_brand"])
        opps.append({"rank": i, "from_brand": p["from_brand"], "to_brand": p["to_brand"],
                     "headline": f"{p['from_brand']} shoppers are {p['affinity_lift']}x as likely to shop {p['to_brand']}",
                     "evidence": [f"Affinity lift {p['affinity_lift']}", f"Audience {d['audience_size']}"],
                     "why_it_matters": "Mock run (no AI call).", "risks": ["Mock output"], "confidence": "medium"})
        options = [(ch, disc, engine.check_policy(p["from_brand"], p["to_brand"], ch, disc))
                   for ch in ("email", "whatsapp") for disc in (0.0, 0.10)]
        options = [o for o in options if o[2]["passed"]]
        ch, disc, best = max(options, key=lambda o: o[2]["estimate_for_eligible"]["expected_incremental_gross_margin_aed"])
        est = best["estimate_for_eligible"]
        actions.append({"from_brand": p["from_brand"], "to_brand": p["to_brand"], "channel": ch, "discount": disc,
                        "expected_incremental_gross_margin_aed": est["expected_incremental_gross_margin_aed"],
                        "eligible_audience": best["eligible_audience"],
                        "control_group_share": est["control_group"]["holdout_share"],
                        "measurable": est["control_group"]["measurable"], "rationale": "Mock run (no AI call).",
                        "alternatives": [], "message_en": f"Hi {{first_name}}, discover {p['to_brand']}.",
                        "message_ar": f"مرحباً {{first_name}}، اكتشف {p['to_brand']}."})
    return {"analyst": {"output": {"opportunities": opps}, "trace": [], "usage": {}},
            "strategist": [{"output": a, "trace": [], "usage": {}} for a in actions]}


# ---------------------------------------------------------------- checks
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers(obj) -> set[float]:
    s = json.dumps(obj, default=str, ensure_ascii=False)
    out = set()
    for m in NUM.findall(s):
        try:
            out.add(round(float(m.replace(",", "")), 2))
        except ValueError:
            pass
    return out


def faithfulness(agent_result: dict) -> dict:
    """Every number in the agent's output should appear in (or be a simple % form of) a tool result."""
    tool_nums = _numbers([t["output"] for t in agent_result["trace"]])
    allowed = tool_nums | {round(x * 100, 2) for x in tool_nums} | {round(x / 100, 4) for x in tool_nums} | {1, 2, 3, 0}
    stated = _numbers(agent_result["output"])
    unsupported = sorted(n for n in stated if not any(abs(n - a) <= max(0.01, abs(a) * 0.01) for a in allowed))
    return {"numbers_stated": len(stated), "unsupported": unsupported, "faithful": not unsupported}


def run_cycle(engine: Engine, mock: bool | None = None, progress: bool = False) -> dict:
    mock = os.environ.get("DESK_MOCK") == "1" if mock is None else mock
    t0 = time.time()
    if mock:
        cycle = _mock_cycle(engine)
    else:
        if progress:
            print("Analyst agent: looking for this week's cross-brand opportunities…", flush=True)
        analyst = run_agent(engine, ANALYST_SYSTEM,
                            f"It is {dt.date.today():%d %B %Y}. Find this week's top {TOP_N} cross-brand opportunities.",
                            ["brand_pair_stats", "pair_detail"], progress=progress)
        strategist = []
        for o in analyst["output"]["opportunities"][:TOP_N]:
            if progress:
                print(f"Strategist agent: designing the action for {o['from_brand']} -> {o['to_brand']}…", flush=True)
            strategist.append(run_agent(engine, STRATEGIST_SYSTEM,
                                        "Design the action for this opportunity:\n" + json.dumps(o, ensure_ascii=False),
                                        ["pair_detail", "compare_options", "check_policy"], progress=progress))
        cycle = {"analyst": analyst, "strategist": strategist}
    cycle["meta"] = {"created": dt.datetime.now().isoformat(timespec="seconds"), "model": "mock" if mock else MODEL,
                     "seconds": round(time.time() - t0, 1),
                     "checks": {"analyst": faithfulness(cycle["analyst"]) if not mock else None,
                                "strategist": [faithfulness(s) for s in cycle["strategist"]] if not mock else None}}
    return cycle


def save_run(cycle: dict, name: str = "latest") -> Path:
    RUNS.mkdir(exist_ok=True)
    path = RUNS / f"{name}.json"
    path.write_text(json.dumps(cycle, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return path


if __name__ == "__main__":
    from .synthetic import generate
    print("Loading the synthetic customer base…", flush=True)
    eng = Engine(generate())
    c = run_cycle(eng, progress=True)
    p = save_run(c, "mock" if c["meta"]["model"] == "mock" else "latest")
    ch = c["meta"]["checks"]
    print(f"\nDone in {c['meta']['seconds']}s.")
    if ch["analyst"]:
        bad = ch["analyst"]["unsupported"] + [n for s in ch["strategist"] for n in s["unsupported"]]
        print("Number check: " + ("all numbers trace back to the engine ✅" if not bad else f"numbers not found in tool results ⚠️ {bad}"))
    for s in c["strategist"]:
        o = s["output"]
        print(f"- {o['from_brand']} -> {o['to_brand']}: {o['channel']}, {o['discount']:.0%} off, "
              f"AED {o['expected_incremental_gross_margin_aed']:,.0f} expected, control {o['control_group_share']:.0%}")
    print("saved", p)
