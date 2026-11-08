"""Customer Opportunity Desk: the manager's inbox.

    streamlit run app/streamlit_app.py

Live runs call Claude on the owner's API key (spend capped in the Anthropic console). If a live run
fails or is unavailable, the desk shows the last saved run, clearly labelled.
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from desk import engine as E  # noqa: E402
from desk.agents import RUNS, run_cycle, save_run  # noqa: E402
from desk.synthetic import TRUTH, generate  # noqa: E402

MAX_LIVE_RUNS_PER_SESSION = 2
CHANNEL_NAMES = {"email": "Email", "sms": "SMS", "app_push": "App push", "whatsapp": "WhatsApp"}

st.set_page_config(page_title="Customer Opportunity Desk", page_icon="🧭", layout="wide")


@st.cache_resource(show_spinner="Loading the customer base…")
def get_engine():
    return E.Engine(generate())


def load_saved():
    for name in ("latest", "mock"):
        p = RUNS / f"{name}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8")), name
    return None, None


eng = get_engine()
ss = st.session_state
ss.setdefault("decisions", {})      # key -> {"status", "reason", "action", "results", "lift"}
ss.setdefault("live_runs", 0)
if "cycle" not in ss:
    ss.cycle, ss.source = load_saved()

# ------------------------------------------------------------------ header
left, right = st.columns([3, 1])
with left:
    st.title("Customer Opportunity Desk")
    st.caption("Al Waha Retail (fictional) · cross-brand growth · **all data is synthetic**")
with right:
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    can_run = has_key and ss.live_runs < MAX_LIVE_RUNS_PER_SESSION
    if st.button("▶ Run the agents live", disabled=not can_run, use_container_width=True,
                 help="Runs the Analyst and Strategist agents on Claude now (about 1–3 minutes)."):
        with st.spinner("Agents are working: analysing brand pairs, then designing actions…"):
            try:
                ss.cycle = run_cycle(eng, mock=False)
                save_run(ss.cycle, "latest")
                ss.source, ss.decisions = "live", {}
                ss.live_runs += 1
            except Exception as e:  # fall back to the saved run, never a broken screen
                st.warning(f"Live run unavailable ({type(e).__name__}). Showing the last saved run instead.")
    if not has_key:
        st.caption("Live runs are off here; showing a saved run.")
    elif ss.live_runs >= MAX_LIVE_RUNS_PER_SESSION:
        st.caption("Live-run limit for this session reached.")

if not ss.cycle:
    st.error("No agent run available yet. Run `python -m desk.agents` once to create one.")
    st.stop()

meta = ss.cycle["meta"]
label = {"live": "Live run just now", "latest": "Saved run", "mock": "Offline sample run (no AI)"}[ss.source]
st.info(f"**{label}** · created {meta['created']} · model {meta['model']} · {meta['seconds']}s")

opps = ss.cycle["analyst"]["output"]["opportunities"]
acts = {(s["output"]["from_brand"], s["output"]["to_brand"]): s for s in ss.cycle["strategist"]}

tab_inbox, tab_results, tab_about = st.tabs(["📥 This week", "📈 Results", "ℹ️ How it works"])

# ------------------------------------------------------------------ inbox
with tab_inbox:
    col_list, col_detail = st.columns([1, 2.2], gap="large")
    with col_list:
        st.subheader("Top opportunities")
        keys = []
        for o in opps:
            key = f"{o['from_brand']}→{o['to_brand']}"
            keys.append(key)
            a = acts.get((o["from_brand"], o["to_brand"]), {}).get("output", {})
            status = ss.decisions.get(key, {}).get("status", "To review")
            with st.container(border=True):
                st.markdown(f"**#{o['rank']} {o['from_brand']} → {o['to_brand']}**")
                st.caption(o["headline"])
                if a:
                    st.caption(f"Expected +AED {a.get('expected_incremental_gross_margin_aed', 0):,.0f} margin · "
                               f"{'measurable' if a.get('measurable') else 'directional'} · **{status}**")
                if st.button("Open", key=f"open_{key}", use_container_width=True,
                             type="primary" if ss.get("sel", keys[0]) == key else "secondary"):
                    ss.sel = key
                    st.rerun()
        sel = ss.get("sel") if ss.get("sel") in keys else keys[0]

    with col_detail:
        o = opps[keys.index(sel)]
        s = acts.get((o["from_brand"], o["to_brand"]))
        a = s["output"] if s else None
        st.subheader(f"{o['from_brand']} → {o['to_brand']}")
        st.markdown(f"**{o['headline']}**")

        with st.container(border=True):
            st.markdown("**Evidence** · from the Analyst agent")
            for ev in o.get("evidence", []):
                st.markdown(f"- {ev}")
            st.markdown(f"**Why it matters:** {o.get('why_it_matters', '')}")
            if o.get("risks"):
                st.markdown("**Risks:** " + "; ".join(o["risks"]))
            st.caption(f"Confidence: {o.get('confidence', 'n/a')}")

        if a:
            decided = ss.decisions.get(sel)
            st.markdown("#### Proposed action · from the Strategist agent")
            c1, c2, c3 = st.columns(3)
            channel = c1.selectbox("Channel", list(CHANNEL_NAMES), index=list(CHANNEL_NAMES).index(a["channel"]),
                                   format_func=CHANNEL_NAMES.get, key=f"ch_{sel}", disabled=bool(decided))
            discount = c2.select_slider("Discount", [0.0, 0.05, 0.10, 0.15], value=float(a["discount"]),
                                        format_func=lambda d: f"{d:.0%}", key=f"d_{sel}", disabled=bool(decided))
            edited = channel != a["channel"] or abs(discount - float(a["discount"])) > 1e-9
            # guardrails + estimate are recomputed by code for whatever is on screen (free, deterministic)
            pol = eng.check_policy(o["from_brand"], o["to_brand"], channel, discount)
            est = pol["estimate_for_eligible"] or {}
            c3.metric("Expected incremental margin", f"AED {est.get('expected_incremental_gross_margin_aed', 0):,.0f}",
                      help="Planning estimate from assumptions. The real number comes from the control group.")
            st.caption(a.get("rationale", "") + (" · *Edited by you: estimate recalculated.*" if edited else ""))

            m1, m2, m3 = st.columns(3)
            cg = est.get("control_group", {})
            m1.metric("Eligible audience", f"{pol['eligible_audience']:,}")
            m1.caption(f"of {pol['audience_before_rules']:,} before consent and contact rules")
            m2.metric("Control group", f"{cg.get('holdout_share', 0):.0%}")
            m2.caption("measurable on its own" if cg.get("measurable") else "too small to read alone: counts toward the programme result")
            m3.metric("Expected discount cost", f"AED {est.get('expected_discount_cost_aed', 0):,.0f}")

            with st.expander("Guardrail checks (enforced by code)", expanded=not pol["passed"]):
                st.markdown("✅ All rules passed" if pol["passed"] else "⛔ " + "; ".join(pol["violations"]))
                st.dataframe(pd.DataFrame([{"Rule": k.replace("_", " "), "Customers excluded": v}
                                           for k, v in pol["excluded_by_rule"].items()]), hide_index=True)

            t1, t2 = st.columns(2)
            msg_en = t1.text_area("Message (English)", a.get("message_en", ""), key=f"en_{sel}", disabled=bool(decided))
            msg_ar = t2.text_area("الرسالة (العربية)", a.get("message_ar", ""), key=f"ar_{sel}", disabled=bool(decided))

            if a.get("alternatives"):
                with st.expander("Options the Strategist considered"):
                    st.dataframe(pd.DataFrame(a["alternatives"]), hide_index=True)

            if not decided:
                b1, b2, b3 = st.columns([1, 1, 2])
                reason = b3.text_input("Reason (if rejecting)", key=f"why_{sel}", placeholder="e.g. Glow has a launch that week")
                if b1.button("✅ Approve & send", type="primary", disabled=not pol["passed"], key=f"ok_{sel}"):
                    act = E.Action(o["from_brand"], o["to_brand"], channel, discount,
                                   holdout_share=cg.get("holdout_share", E.POLICY["global_holdout"]),
                                   message_en=msg_en, message_ar=msg_ar)
                    res = E.simulate_send(eng, act)
                    ss.decisions[sel] = {"status": "Approved (edited)" if edited else "Approved", "action": act,
                                         "results": res, "lift": E.measure_lift(res), "measurable": cg.get("measurable")}
                    st.rerun()
                if b2.button("✖ Reject", key=f"no_{sel}"):
                    ss.decisions[sel] = {"status": "Rejected", "reason": reason or "no reason given"}
                    st.rerun()
            else:
                st.success(f"{decided['status']}" + (f": {decided.get('reason')}" if decided.get("reason") else
                                                     ". Simulated send done; see the Results tab."))

            with st.expander("How the agents got here"):
                for who, res in (("Analyst", ss.cycle["analyst"]), ("Strategist", s)):
                    st.markdown(f"**{who}**: {len(res['trace'])} tool calls")
                    for i, t in enumerate(res["trace"], 1):
                        st.markdown(f"{i}. `{t['tool']}({json.dumps(t['input'])})`")
                        st.json(t["output"], expanded=False)
                checks = meta.get("checks") or {}
                if checks.get("analyst"):
                    st.markdown("**Number check:** every number the agents wrote should come from a tool result.")
                    allc = [checks["analyst"]] + list(checks.get("strategist") or [])
                    bad = [n for c in allc for n in c["unsupported"]]
                    st.markdown("✅ No invented numbers" if not bad else f"⚠️ Numbers not found in tool results: {bad}")

# ------------------------------------------------------------------ results
with tab_results:
    sent = {k: v for k, v in ss.decisions.items() if "lift" in v}
    if not sent:
        st.info("Approve an action in the inbox to see its simulated result here.")
    else:
        st.subheader("Campaign results (simulated, 90 days)")
        rows = []
        for k, v in sent.items():
            L = v["lift"]
            rows.append({"Campaign": k, "Channel": CHANNEL_NAMES[v["action"].channel], "Discount": f"{v['action'].discount:.0%}",
                         "Contacted": L["treated"], "Control": L["holdout"],
                         "Conversion contacted": f"{L['conversion_treated']:.1%}", "Conversion control": f"{L['conversion_holdout']:.1%}",
                         "Incremental margin (AED)": f"{L['incremental_gross_margin_aed']:,.0f}",
                         "95% range (AED)": f"{L['ci95_total_aed'][0]:,.0f} to {L['ci95_total_aed'][1]:,.0f}",
                         "Read as": ("significant" if L["significant"] else "not yet significant")
                         + ("" if v["measurable"] else " (planned as directional)")})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        prog = E.measure_programme([v["results"] for v in sent.values()])
        st.subheader("Programme vs global holdout")
        p1, p2, p3 = st.columns(3)
        p1.metric("Incremental gross margin", f"AED {prog['incremental_gross_margin_aed']:,.0f}")
        p2.metric("95% range", f"{prog['ci95_total_aed'][0]:,.0f} to {prog['ci95_total_aed'][1]:,.0f}")
        p3.metric("Result", "significant" if prog["significant"] else "not yet significant", delta_color="off")
        st.caption("Single campaigns are often too small to read alone. The programme pools every contacted customer "
                   "against every control customer: the 10% global holdout plus any campaign-level control.")

        with st.expander("Evaluation: did the measurement find the truth? (only possible with synthetic data)"):
            ev = []
            for k, v in sent.items():
                truth = E.true_incremental_margin_per_customer(eng, v["action"]) * v["lift"]["treated"]
                lo, hi = v["lift"]["ci95_total_aed"]
                ev.append({"Campaign": k, "Measured (AED)": round(v["lift"]["incremental_gross_margin_aed"]),
                           "True planted effect (AED)": round(truth), "Range contains truth": lo <= truth <= hi})
            st.dataframe(pd.DataFrame(ev), hide_index=True)
            planted = set(TRUTH.true_pairs())
            found = sum((o["from_brand"], o["to_brand"]) in planted for o in opps)
            st.markdown(f"Analyst's top {len(opps)}: **{found} of {len(opps)}** are brand affinities planted in the data.")

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown("""
**What this is.** A decision-support prototype for a customer growth manager at a multi-brand retail group.
Two AI agents find cross-brand opportunities and draft actions; code does every calculation and enforces the rules;
the manager approves; the send is simulated and measured against a control group in **incremental gross margin**.

**Who does what**
- *Analyst agent*: ranks brand pairs and explains the evidence. Never calculates.
- *Strategist agent*: designs the action (channel, offer, message in English and Arabic), message first, discount only if it pays.
- *Engine (code)*: affinity, audience, margin estimate, consent and contact rules, control-group sizing, measurement.
- *You*: approve, edit or reject. Nothing is sent without approval.

**Honesty notes.** The data is synthetic, with known answers planted so the system can be tested. Margin estimates
before sending are planning assumptions; only the control-group comparison counts as a result.
""")
