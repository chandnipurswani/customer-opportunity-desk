"""Customer Opportunity Desk: the manager's inbox.

    streamlit run app/streamlit_app.py

Live runs call Claude on the owner's API key (spend capped in the Anthropic console). If a live run
fails or is unavailable, the desk shows the last saved run, clearly labelled.
"""
import html
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
CHANNEL = {"email": "Email", "sms": "SMS", "app_push": "App notification", "whatsapp": "WhatsApp"}

st.set_page_config(page_title="Customer Opportunity Desk", page_icon="🧭", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Arabic:wght@400;500&display=swap');
:root { --bg:#F2F4F7; --panel:#FFFFFF; --ink:#172033; --muted:#5E6A7D; --rule:#DCE1E8;
        --teal:#0E7C74; --teal-soft:#E3F2F0; --gold:#B7790B; --gold-soft:#FBF3E2; --red:#B42318; --red-soft:#FDECEA; }
html, body, [class*="css"], .stApp, button, input, textarea, select { font-family:'IBM Plex Sans', sans-serif !important; color:var(--ink); }
.stApp { background:var(--bg); }
#MainMenu, header[data-testid="stHeader"], footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
.block-container { padding-top:1.6rem; max-width:1280px; }
h1, h2, h3 { letter-spacing:-0.01em; }
.topbar { display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:0.4rem; }
.topbar h1 { font-size:1.65rem; font-weight:600; margin:0; }
.topbar p { margin:0.2rem 0 0; color:var(--muted); font-size:0.92rem; }
.runline { color:var(--muted); font-size:0.85rem; text-align:right; }
.runline b { color:var(--ink); font-weight:500; }
.weekline { background:var(--panel); border:1px solid var(--rule); border-radius:10px; padding:0.9rem 1.1rem; margin:0.6rem 0 1.2rem;
            font-size:1rem; }
.weekline .gold { color:var(--gold); font-weight:600; }
.ticket { background:var(--panel); border:1px solid var(--rule); border-radius:10px; padding:0.9rem 1rem 0.7rem; margin-bottom:0.35rem; }
.ticket.sel { border-color:var(--teal); box-shadow:inset 4px 0 0 var(--teal); }
.ticket .pair { font-weight:600; font-size:1.02rem; }
.ticket .pair span { color:var(--muted); font-weight:400; margin:0 0.35rem; }
.ticket .line { color:var(--muted); font-size:0.88rem; margin:0.25rem 0 0.5rem; line-height:1.4; }
.ticket .foot { display:flex; justify-content:space-between; align-items:center; font-size:0.85rem; }
.ticket .money { color:var(--gold); font-weight:600; }
.chip { display:inline-block; padding:0.12rem 0.55rem; border-radius:999px; font-size:0.78rem; font-weight:500; background:#EEF1F5; color:var(--muted); }
.chip.ok { background:var(--teal-soft); color:var(--teal); } .chip.no { background:var(--red-soft); color:var(--red); }
.pairhero { display:flex; align-items:center; gap:0; background:var(--panel); border:1px solid var(--rule); border-radius:12px; padding:1.2rem 1.4rem; }
.store { flex:0 0 32%; border:1.5px solid var(--ink); border-radius:10px; padding:0.8rem 1rem; }
.store .name { font-size:1.35rem; font-weight:600; }
.store .what { color:var(--muted); font-size:0.86rem; }
.store.to { border-color:var(--teal); background:var(--teal-soft); }
.bridge { flex:1; text-align:center; position:relative; padding:0 0.6rem; }
.bridge .x { font-size:1.6rem; font-weight:600; color:var(--teal); line-height:1; }
.bridge .cap { color:var(--muted); font-size:0.82rem; margin-top:0.2rem; }
.bridge .arrow { height:2px; background:var(--teal); margin:0.5rem 0 0; position:relative; }
.bridge .arrow:after { content:''; position:absolute; right:-2px; top:-5px; border-left:9px solid var(--teal); border-top:6px solid transparent; border-bottom:6px solid transparent; }
.headline { font-size:1.08rem; margin:0.9rem 0 0.2rem; }
.sect { font-size:1.02rem; font-weight:600; margin:1.3rem 0 0.4rem; }
.panel { background:var(--panel); border:1px solid var(--rule); border-radius:10px; padding:0.9rem 1.1rem; }
.panel ul { margin:0.2rem 0 0.4rem 1.1rem; padding:0; } .panel li { margin:0.15rem 0; }
.muted { color:var(--muted); font-size:0.88rem; }
.figs { display:grid; grid-template-columns:repeat(4, 1fr); gap:0.6rem; margin-top:0.6rem; }
.fig { background:var(--panel); border:1px solid var(--rule); border-radius:10px; padding:0.75rem 0.9rem; }
.fig .v { font-size:1.35rem; font-weight:600; font-variant-numeric:tabular-nums; }
.fig .k { color:var(--muted); font-size:0.82rem; margin-top:0.1rem; line-height:1.35; }
.fig.main { background:var(--gold-soft); border-color:#EBD5A7; } .fig.main .v { color:var(--gold); }
.rule-row { display:flex; justify-content:space-between; padding:0.3rem 0; border-bottom:1px solid #EEF1F4; font-size:0.9rem; }
.rule-row:last-child { border-bottom:none; }
textarea[aria-label="Arabic message"] { direction:rtl; text-align:right; font-family:'IBM Plex Sans Arabic', sans-serif !important; font-size:1rem !important; }
.stButton > button { border-radius:8px; font-weight:500; border:1px solid var(--rule); }
.stButton > button[kind="primary"] { background:var(--teal); border-color:var(--teal); color:#fff; }
.stButton > button[kind="primary"]:hover { background:#0B655F; border-color:#0B655F; }
.stTabs [data-baseweb="tab-list"] { gap:1.4rem; border-bottom:1px solid var(--rule); }
.stTabs [data-baseweb="tab"] { font-weight:500; padding:0.5rem 0; }
.stTabs [aria-selected="true"] { color:var(--teal) !important; }
.stTabs [data-baseweb="tab-highlight"] { background:var(--teal) !important; }
.done { background:var(--teal-soft); color:var(--teal); border-radius:10px; padding:0.8rem 1rem; font-weight:500; }
.done.no { background:var(--red-soft); color:var(--red); }
@media (max-width: 900px) { .figs { grid-template-columns:repeat(2, 1fr); } .pairhero { flex-direction:column; gap:0.8rem; } .store { flex:auto; width:100%; } }
</style>
""", unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def aed(x) -> str:
    return f"AED {x:,.0f}" if x is not None else "n/a"


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
ss.setdefault("decisions", {})
ss.setdefault("live_runs", 0)
if "cycle" not in ss:
    ss.cycle, ss.source = load_saved()
known = set(eng.brands.index)

# ------------------------------------------------------------------ top bar
left, right = st.columns([3, 1.2])
with left:
    st.markdown("""<div class="topbar"><div><h1>Customer Opportunity Desk</h1>
    <p>Weekly cross-store opportunities for Al Waha Retail, a fictional group of six stores. All customer data is synthetic.</p>
    </div></div>""", unsafe_allow_html=True)
with right:
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    can_run = has_key and ss.live_runs < MAX_LIVE_RUNS_PER_SESSION
    if st.button("Run the agents now", disabled=not can_run, use_container_width=True,
                 help="Runs the Analyst and Strategist agents on Claude (about 1 minute)."):
        with st.spinner("The Analyst is ranking store pairs, then the Strategist designs each action…"):
            try:
                ss.cycle = run_cycle(eng, mock=False)
                save_run(ss.cycle, "latest")
                ss.source, ss.decisions = "live", {}
                ss.live_runs += 1
            except Exception as e:
                st.warning(f"The live run didn't complete ({type(e).__name__}). Showing the last saved run instead.")
    meta = (ss.cycle or {}).get("meta", {})
    src = {"live": "Live run, just now", "latest": "Saved agent run", "mock": "Offline sample (no AI)"}.get(ss.source, "")
    note = "" if has_key else "<br>Live runs are switched off here."
    if has_key and ss.live_runs >= MAX_LIVE_RUNS_PER_SESSION:
        note = "<br>Live-run limit for this visit reached."
    st.markdown(f'<div class="runline"><b>{esc(src)}</b><br>{esc(meta.get("created", "")).replace("T", " ")}{note}</div>',
                unsafe_allow_html=True)

if not ss.cycle:
    st.error("There's no agent run yet. Run `python -m desk.agents` once to create one.")
    st.stop()

opps = [o for o in ss.cycle["analyst"]["output"]["opportunities"] if o["from_brand"] in known and o["to_brand"] in known]
acts = {(s["output"]["from_brand"], s["output"]["to_brand"]): s for s in ss.cycle["strategist"]}
if not opps:
    st.warning("The saved run uses an older version of the store names. Run the agents again to refresh it.")
    st.stop()

total = sum(acts[(o["from_brand"], o["to_brand"])]["output"].get("expected_incremental_gross_margin_aed", 0)
            for o in opps if (o["from_brand"], o["to_brand"]) in acts)
st.markdown(f'<div class="weekline">This week the agents found <b>{len(opps)} opportunities</b>. If you approve all of them, '
            f'the expected extra gross margin is <span class="gold">{aed(total)}</span> over 90 days.</div>',
            unsafe_allow_html=True)

tab_inbox, tab_results, tab_about = st.tabs(["This week", "Results", "How it works"])

# ------------------------------------------------------------------ inbox
with tab_inbox:
    col_list, col_detail = st.columns([1, 2.3], gap="large")
    keys = [f"{o['from_brand']}→{o['to_brand']}" for o in opps]
    sel = ss.get("sel") if ss.get("sel") in keys else keys[0]
    with col_list:
        for o, key in zip(opps, keys):
            a = acts.get((o["from_brand"], o["to_brand"]), {}).get("output", {})
            dec = ss.decisions.get(key, {}).get("status")
            chip = (f'<span class="chip ok">{esc(dec)}</span>' if dec and dec.startswith("Approved") else
                    f'<span class="chip no">{esc(dec)}</span>' if dec else '<span class="chip">To review</span>')
            st.markdown(f"""<div class="ticket {'sel' if key == sel else ''}">
              <div class="pair">{esc(o['from_brand'])}<span>to</span>{esc(o['to_brand'])}</div>
              <div class="line">{esc(o['headline'])}</div>
              <div class="foot"><span class="money">+{aed(a.get('expected_incremental_gross_margin_aed'))}</span>{chip}</div>
            </div>""", unsafe_allow_html=True)
            if st.button("Open" if key != sel else "Selected", key=f"open_{key}", use_container_width=True,
                         disabled=key == sel):
                ss.sel = key
                st.rerun()

    with col_detail:
        o = opps[keys.index(sel)]
        s = acts.get((o["from_brand"], o["to_brand"]))
        a = s["output"] if s else None
        d = eng.pair_detail(o["from_brand"], o["to_brand"])
        fb, tb = eng.brands.loc[o["from_brand"]], eng.brands.loc[o["to_brand"]]
        st.markdown(f"""<div class="pairhero">
          <div class="store"><div class="name">{esc(o['from_brand'])}</div><div class="what">{esc(fb.category)}</div></div>
          <div class="bridge"><div class="x">{d['affinity_lift']:.1f}×</div>
            <div class="cap">as likely as other shoppers to also buy at {esc(o['to_brand'])}</div><div class="arrow"></div></div>
          <div class="store to"><div class="name">{esc(o['to_brand'])}</div><div class="what">{esc(tb.category)}</div></div>
        </div>
        <div class="headline">{esc(o['headline'])}</div>""", unsafe_allow_html=True)

        ev = "".join(f"<li>{esc(x)}</li>" for x in o.get("evidence", []))
        risks = "".join(f"<li>{esc(x)}</li>" for x in o.get("risks", []))
        st.markdown(f"""<div class="sect">Why this pair</div><div class="panel">
          <ul>{ev}</ul><div>{esc(o.get('why_it_matters', ''))}</div>
          {f'<div class="muted" style="margin-top:0.5rem">Watch out for</div><ul>{risks}</ul>' if risks else ''}
          <div class="muted">The Analyst's confidence: {esc(o.get('confidence', 'n/a'))}</div></div>""", unsafe_allow_html=True)

        if a:
            decided = ss.decisions.get(sel)
            st.markdown('<div class="sect">Proposed action</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            channel = c1.selectbox("Channel", list(CHANNEL), index=list(CHANNEL).index(a["channel"]),
                                   format_func=CHANNEL.get, key=f"ch_{sel}", disabled=bool(decided))
            discount = c2.select_slider("Discount on the first order", [0.0, 0.05, 0.10, 0.15], value=float(a["discount"]),
                                        format_func=lambda x: "None" if x == 0 else f"{x:.0%} off", key=f"d_{sel}",
                                        disabled=bool(decided))
            edited = channel != a["channel"] or abs(discount - float(a["discount"])) > 1e-9
            pol = eng.check_policy(o["from_brand"], o["to_brand"], channel, discount)
            est = pol["estimate_for_eligible"] or {}
            cg = est.get("control_group", {})
            st.markdown(f"""<div class="figs">
              <div class="fig main"><div class="v">{aed(est.get('expected_incremental_gross_margin_aed'))}</div>
                <div class="k">Expected extra gross margin, after discount cost (planning estimate)</div></div>
              <div class="fig"><div class="v">{pol['eligible_audience']:,}</div>
                <div class="k">Customers reached, of {pol['audience_before_rules']:,} before the rules</div></div>
              <div class="fig"><div class="v">{cg.get('holdout_share', 0):.0%}</div>
                <div class="k">Held back to measure impact{'' if cg.get('measurable') else '; too few to judge alone'}</div></div>
              <div class="fig"><div class="v">{aed(est.get('expected_discount_cost_aed'))}</div>
                <div class="k">Expected cost of the discount</div></div>
            </div>""", unsafe_allow_html=True)
            why = a.get("rationale", "")
            st.markdown(f'<div class="panel" style="margin-top:0.6rem"><span class="muted">Strategist\'s reasoning</span><br>'
                        f'{esc(why)}{"<br><i>You changed the action, so the figures above were recalculated.</i>" if edited else ""}'
                        f'</div>', unsafe_allow_html=True)

            rows = "".join(f'<div class="rule-row"><span>{esc(k)}</span><span>{v:,} customers left out</span></div>' for k, v in [
                ("No permission for this channel", pol["excluded_by_rule"]["no_consent"]),
                ("Contacted in the last 7 days", pol["excluded_by_rule"]["contacted_recently"]),
                ("Already contacted 4+ times this month", pol["excluded_by_rule"]["contact_cap"]),
                ("In the permanent 10% comparison group", pol["excluded_by_rule"]["global_holdout_not_contacted"])])
            status = ('<span class="chip ok">All rules pass</span>' if pol["passed"] else
                      f'<span class="chip no">Blocked: {esc("; ".join(pol["violations"]))}</span>')
            st.markdown(f'<div class="sect">Rules check</div><div class="panel">{status}'
                        f'<div style="margin-top:0.5rem">{rows}</div><div class="muted" style="margin-top:0.4rem">'
                        f'Checked by code, not by the AI. Discounts above 15% and campaigns over the AED 50,000 discount '
                        f'budget are blocked.</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="sect">Customer message</div>', unsafe_allow_html=True)
            t1, t2 = st.columns(2)
            msg_en = t1.text_area("English message", a.get("message_en", ""), key=f"en_{sel}", disabled=bool(decided), height=110)
            msg_ar = t2.text_area("Arabic message", a.get("message_ar", ""), key=f"ar_{sel}", disabled=bool(decided), height=110)

            if a.get("alternatives"):
                with st.expander("Other options the Strategist compared"):
                    alt = pd.DataFrame(a["alternatives"])
                    alt["channel"] = alt["channel"].map(lambda c: CHANNEL.get(c, c))
                    alt["discount"] = alt["discount"].map(lambda x: "None" if x == 0 else f"{x:.0%}")
                    alt = alt.rename(columns={"channel": "Channel", "discount": "Discount",
                                              "expected_incremental_gross_margin_aed": "Expected extra margin (AED)",
                                              "gap_vs_best_aed": "Less than the chosen option (AED)", "why_not": "Why not"})
                    st.dataframe(alt, hide_index=True, use_container_width=True)

            st.markdown('<div class="sect">Your decision</div>', unsafe_allow_html=True)
            if not decided:
                b1, b2, b3 = st.columns([1.1, 0.9, 2.2])
                reason = b3.text_input("Reason for rejecting (optional)", key=f"why_{sel}",
                                       placeholder="For example: Beauty has a launch that week", label_visibility="collapsed")
                if b1.button("Approve and send", type="primary", disabled=not pol["passed"], key=f"ok_{sel}",
                             use_container_width=True):
                    act = E.Action(o["from_brand"], o["to_brand"], channel, discount,
                                   holdout_share=cg.get("holdout_share", E.POLICY["global_holdout"]),
                                   message_en=msg_en, message_ar=msg_ar)
                    res = E.simulate_send(eng, act)
                    ss.decisions[sel] = {"status": "Approved, edited" if edited else "Approved", "action": act,
                                         "results": res, "lift": E.measure_lift(res), "measurable": cg.get("measurable")}
                    st.rerun()
                if b2.button("Reject", key=f"no_{sel}", use_container_width=True):
                    ss.decisions[sel] = {"status": "Rejected", "reason": reason or "no reason given"}
                    st.rerun()
            elif decided["status"].startswith("Approved"):
                st.markdown('<div class="done">Approved. The send was simulated; see the Results tab for what happened '
                            'over 90 days.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="done no">Rejected: {esc(decided.get("reason"))}. The reason is logged for the '
                            f'agents.</div>', unsafe_allow_html=True)

            with st.expander("How the agents got here"):
                for who, res in (("Analyst", ss.cycle["analyst"]), ("Strategist", s)):
                    st.markdown(f"**{who} agent** made {len(res['trace'])} calls to the calculation engine:")
                    for i, t in enumerate(res["trace"], 1):
                        args = ", ".join(f"{k}={v}" for k, v in t["input"].items() if k != "options")
                        st.markdown(f"{i}. `{t['tool']}({args})`")
                        st.json(t["output"], expanded=False)
                checks = meta.get("checks") or {}
                if checks.get("analyst"):
                    bad = [n for c in [checks["analyst"], *(checks.get("strategist") or [])] for n in c["unsupported"]]
                    st.markdown("**Number check.** Every number the agents wrote should come from the engine. " +
                                ("All of them do." if not bad else f"These don't: {bad}."))

# ------------------------------------------------------------------ results
with tab_results:
    sent = {k: v for k, v in ss.decisions.items() if "lift" in v}
    if not sent:
        st.markdown('<div class="panel">Nothing has been sent yet. Approve an action under <b>This week</b> and its '
                    'simulated 90-day result appears here.</div>', unsafe_allow_html=True)
    else:
        prog = E.measure_programme([v["results"] for v in sent.values()])
        st.markdown(f"""<div class="sect">Overall result, all approved campaigns</div><div class="figs">
          <div class="fig main"><div class="v">{aed(prog['incremental_gross_margin_aed'])}</div>
            <div class="k">Extra gross margin caused by the campaigns</div></div>
          <div class="fig"><div class="v">{prog['ci95_total_aed'][0]:,.0f} to {prog['ci95_total_aed'][1]:,.0f}</div>
            <div class="k">Likely range, AED (95% confidence)</div></div>
          <div class="fig"><div class="v">{prog['conversion_treated']:.1%} vs {prog['conversion_holdout']:.1%}</div>
            <div class="k">Bought at the new store: contacted vs held back</div></div>
          <div class="fig"><div class="v">{'Yes' if prog['significant'] else 'Not yet'}</div>
            <div class="k">Clear evidence the campaigns worked</div></div>
        </div>""", unsafe_allow_html=True)
        rows = []
        for k, v in sent.items():
            L = v["lift"]
            rows.append({"Campaign": k.replace("→", " to "), "Channel": CHANNEL[v["action"].channel],
                         "Discount": "None" if v["action"].discount == 0 else f"{v['action'].discount:.0%}",
                         "Contacted": L["treated"], "Held back": L["holdout"],
                         "Bought, contacted": f"{L['conversion_treated']:.1%}", "Bought, held back": f"{L['conversion_holdout']:.1%}",
                         "Extra margin (AED)": f"{L['incremental_gross_margin_aed']:,.0f}",
                         "Likely range (AED)": f"{L['ci95_total_aed'][0]:,.0f} to {L['ci95_total_aed'][1]:,.0f}",
                         "Verdict": "Clear result" if L["significant"] else
                         ("Not clear yet" if v["measurable"] else "Too small to judge alone")})
        st.markdown('<div class="sect">Each campaign</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.markdown('<div class="muted">A single campaign often reaches too few people to judge on its own. The overall '
                    'result pools everyone contacted against everyone held back, including the permanent 10% comparison '
                    'group.</div>', unsafe_allow_html=True)

        with st.expander("Test: did the measurement find the true effect? (possible only because the data is synthetic)"):
            ev = []
            for k, v in sent.items():
                truth = E.true_incremental_margin_per_customer(eng, v["action"]) * v["lift"]["treated"]
                lo, hi = v["lift"]["ci95_total_aed"]
                ev.append({"Campaign": k.replace("→", " to "), "Measured (AED)": round(v["lift"]["incremental_gross_margin_aed"]),
                           "True effect built into the data (AED)": round(truth),
                           "Range contains the truth": "Yes" if lo <= truth <= hi else "No"})
            st.dataframe(pd.DataFrame(ev), hide_index=True, use_container_width=True)
            planted = set(TRUTH.true_pairs())
            found = sum((o["from_brand"], o["to_brand"]) in planted for o in opps)
            st.markdown(f"Of the Analyst's {len(opps)} picks, **{found}** are store pairs we deliberately built into the data.")

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown("""<div class="panel" style="max-width:780px; line-height:1.6">
<p><b>What this is.</b> A working prototype of a weekly tool for the person who grows customer value across a group of stores.
It looks for shoppers of one store who are likely to start buying at another, proposes how to reach them, and, once
approved, measures whether it worked.</p>
<p><b>Who does what</b></p>
<ul>
<li><b>Analyst agent (AI):</b> ranks store pairs and explains the evidence in plain language. It never calculates.</li>
<li><b>Strategist agent (AI):</b> proposes the channel, any discount, and a message in English and Arabic. It tries
no-discount options first and suggests a discount only if it earns more margin.</li>
<li><b>Calculation engine (code):</b> does every number, applies consent and contact rules, and sizes the group held back
for measurement.</li>
<li><b>You:</b> approve, change or reject. Nothing is sent without your approval.</li>
</ul>
<p><b>What's real and what isn't.</b> The stores and customers are invented, and the sends are simulated. The data has
known patterns built in, so we can test whether the agents find them and whether the measurement recovers the true
effect. Figures before sending are planning estimates; only the comparison with customers held back counts as a result.</p>
</div>""", unsafe_allow_html=True)
