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
    """Most recent saved run that matches the current store names (older runs are skipped, never shown broken)."""
    stores = set(generate.__globals__["B"])
    for name in ("latest", "mock"):
        p = RUNS / f"{name}.json"
        if p.exists():
            c = json.loads(p.read_text(encoding="utf-8"))
            if all(o["from_brand"] in stores for o in c["analyst"]["output"]["opportunities"]):
                return c, name
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

tab_inbox, tab_results, tab_about = st.tabs(["This week", "Performance", "How it works"])

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
                st.markdown('<div class="done">Approved. The send was simulated; see the Performance tab for what happened '
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
TEST, CONTROL = "#0E7C74", "#6B7588"


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def segment_uplift(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for seg, g in df.groupby(col):
        t, c = g[g.treated], g[~g.treated]
        if len(t) < 30 or len(c) < 30:
            continue
        pt, pc = t.converted.mean(), c.converted.mean()
        se = ((pt * (1 - pt)) / len(t) + (pc * (1 - pc)) / len(c)) ** 0.5
        rows.append({"Segment": str(seg), "Uplift (pts)": (pt - pc) * 100, "low": (pt - pc - 1.96 * se) * 100,
                     "high": (pt - pc + 1.96 * se) * 100, "Test CR": pt, "Control CR": pc,
                     "Customers": len(g)})
    return pd.DataFrame(rows)


with tab_results:
    import altair as alt
    sent = {k: v for k, v in ss.decisions.items() if "lift" in v}
    if not sent:
        st.markdown('<div class="panel">No campaigns are live yet. Approve an action under <b>This week</b> to see its '
                    'simulated 90-day performance here.</div>', unsafe_allow_html=True)
    else:
        allres = pd.concat([v["results"] for v in sent.values()], ignore_index=True)
        prog = E.measure_programme([v["results"] for v in sent.values()])
        # margin bridge and lifetime value, per campaign, then summed
        inc_rev = cogs = disc = clv = 0.0
        inc_cust = 0.0
        for v in sent.values():
            L, a = v["lift"], v["action"]
            gm = float(eng.brands.loc[a.to_brand].gross_margin)
            inc_rev += L["incremental_revenue_aed"]
            cogs += L["incremental_revenue_aed"] * (1 - gm)
            disc += L["discount_cost_aed"]
            inc_cust += L["incremental_customers"]
            clv += max(L["incremental_customers"], 0) * eng.store_12m_margin_per_new_customer(a.to_brand)
        send_cost = sum(v["lift"]["send_cost_aed"] for v in sent.values())
        fixed = E.CAMPAIGN_FIXED_COST * len(sent)
        roi = prog["incremental_gross_margin_aed"] / (disc + send_cost + fixed)
        sig = prog["p_value"] < 0.05
        ptxt = "p < 0.01" if prog["p_value"] < 0.01 else f"p = {prog['p_value']:.2f}"
        badge = (f'<span class="chip ok">Significant at 95% ({ptxt})</span>' if sig else
                 f'<span class="chip">Not yet significant (p = {prog["p_value"]:.2f})</span>')

        st.markdown(f"""<div class="sect">Campaign performance: {len(sent)} live campaign{'s' if len(sent) > 1 else ''},
          90 days after send, test vs control</div>
        <div class="figs" style="grid-template-columns:1.25fr 1fr 1fr 1fr 1fr">
          <div class="fig main"><div class="v">{aed(prog['incremental_gross_margin_aed'])}</div>
            <div class="k">Incremental gross margin<br>likely range {prog['ci95_total_aed'][0]:,.0f} to {prog['ci95_total_aed'][1]:,.0f}</div>
            <div style="margin-top:0.4rem">{badge}</div></div>
          <div class="fig"><div class="v">{aed(inc_rev)}</div><div class="k">Incremental revenue</div></div>
          <div class="fig"><div class="v">{f'{roi:.1f}×' if roi else 'n/a'}</div>
            <div class="k">Return on campaign cost (discounts, messaging and setup)</div></div>
          <div class="fig"><div class="v">{pct(prog['cr_test'])} <span style="color:var(--muted);font-weight:400;font-size:1rem">vs {pct(prog['cr_control'])}</span></div>
            <div class="k">Conversion rate, test vs control<br>+{prog['uplift_pp']:.1f} pts uplift</div></div>
          <div class="fig"><div class="v">+{inc_cust:,.0f}</div>
            <div class="k">New cross-store customers<br>worth about {aed(clv)} in projected 12-month margin</div></div>
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="large")
        # ---- cumulative conversion curve
        with c1:
            st.markdown('<div class="sect">Conversion rate over the 90 days</div>', unsafe_allow_html=True)
            days = pd.DataFrame({"day": range(1, 91)})
            curves = []
            for grp, g in (("Test", allres[allres.treated]), ("Control", allres[~allres.treated])):
                cum = [(g.conversion_day <= d).sum() / len(g) for d in days.day]
                curves.append(pd.DataFrame({"Day": days.day, "Conversion rate": cum, "Group": grp}))
            cdf = pd.concat(curves)
            color = alt.Scale(domain=["Test", "Control"], range=[TEST, CONTROL])
            base = alt.Chart(cdf).encode(
                x=alt.X("Day:Q", title="Days since send", scale=alt.Scale(domain=[1, 90]), axis=alt.Axis(grid=False, tickCount=6)),
                y=alt.Y("Conversion rate:Q", title=None, axis=alt.Axis(format="%", gridColor="#E6E9EE", tickCount=4)),
                color=alt.Color("Group:N", scale=color, legend=alt.Legend(orient="top", title=None)))
            lines = base.mark_line(strokeWidth=2.25)
            hover = alt.selection_point(fields=["Day"], nearest=True, on="pointerover", empty=False)
            pts = base.mark_point(size=60, filled=True).encode(
                opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                tooltip=[alt.Tooltip("Group:N"), alt.Tooltip("Day:Q"), alt.Tooltip("Conversion rate:Q", format=".2%")]
            ).add_params(hover)
            ends = base.transform_filter(alt.datum.Day == 90).mark_text(align="left", dx=6, fontSize=12).encode(
                text="Group:N", color=alt.Color("Group:N", scale=color, legend=None))
            st.altair_chart((lines + pts + ends).properties(height=260).configure_view(strokeWidth=0)
                            .configure_axis(labelColor="#5E6A7D", domainColor="#DCE1E8", tickColor="#DCE1E8"),
                            use_container_width=True)
            st.markdown('<div class="muted">Most of the response comes in the first three weeks; the control group shows '
                        'the cross-store buying that would have happened anyway.</div>', unsafe_allow_html=True)

        # ---- margin bridge (waterfall)
        with c2:
            st.markdown('<div class="sect">From incremental revenue to margin</div>', unsafe_allow_html=True)
            # cost of goods is implied so the bridge ends exactly at the measured incremental margin
            cogs = inc_rev - disc - prog["incremental_gross_margin_aed"]
            steps = [("Incremental revenue", inc_rev), ("Cost of goods", -cogs), ("Discounts given", -disc)]
            rows, run = [], 0.0
            for name, val in steps:
                rows.append({"Step": name, "start": run, "end": run + val, "AED": val, "kind": "total" if name.startswith("Incr") else "cost"})
                run += val
            rows.append({"Step": "Incremental gross margin", "start": 0, "end": prog["incremental_gross_margin_aed"],
                         "AED": prog["incremental_gross_margin_aed"], "kind": "result"})
            wf = pd.DataFrame(rows)
            order = list(wf.Step)
            bars = alt.Chart(wf).mark_bar(cornerRadius=3, size=46).encode(
                x=alt.X("Step:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=140)),
                y=alt.Y("start:Q", title=None, axis=alt.Axis(format=",.0f", gridColor="#E6E9EE", tickCount=4)),
                y2="end:Q",
                color=alt.Color("kind:N", scale=alt.Scale(domain=["total", "cost", "result"],
                                                          range=["#9FD3CD", "#C9CED7", "#B7790B"]), legend=None),
                tooltip=[alt.Tooltip("Step:N"), alt.Tooltip("AED:Q", format=",.0f")])
            labels = alt.Chart(wf).mark_text(dy=-8, fontSize=12, color="#172033").encode(
                x=alt.X("Step:N", sort=order), y=alt.Y("max_y:Q"), text=alt.Text("AED:Q", format=",.0f")
            ).transform_calculate(max_y="max(datum.start, datum.end)")
            st.altair_chart((bars + labels).properties(height=260).configure_view(strokeWidth=0)
                            .configure_axis(labelColor="#5E6A7D", domainColor="#DCE1E8", tickColor="#DCE1E8"),
                            use_container_width=True)
            st.markdown(f'<div class="muted">Discounts include those given to customers who would have bought anyway, '
                        f'which is why margin, not revenue, is the measure of success. Messaging cost: {aed(send_cost)}.</div>',
                        unsafe_allow_html=True)

        # ---- uplift by segment
        st.markdown('<div class="sect">Where it worked: conversion uplift by segment</div>', unsafe_allow_html=True)
        dims = {"Customer value": "value_tier", "Language": "language", "City": "city", "Campaign": "campaign"}
        dim = st.radio("Break down by", list(dims), horizontal=True, label_visibility="collapsed")
        seg = segment_uplift(allres, dims[dim])
        if seg.empty:
            st.markdown('<div class="muted">Not enough customers per segment yet.</div>', unsafe_allow_html=True)
        else:
            if dim == "Customer value":
                seg["Segment"] = pd.Categorical(seg.Segment, ["High value", "Mid value", "Low value"], ordered=True)
                seg = seg.sort_values("Segment")
            sb = alt.Chart(seg).encode(y=alt.Y("Segment:N", title=None, sort=list(seg.Segment.astype(str))))
            err = sb.mark_rule(color="#8A94A6", strokeWidth=1.5).encode(x=alt.X("low:Q", title="Uplift in conversion rate, percentage points (with 95% range)", axis=alt.Axis(tickCount=6, format=".1f")), x2="high:Q")
            dot = sb.mark_point(filled=True, size=110, color=TEST).encode(
                x="Uplift (pts):Q",
                tooltip=[alt.Tooltip("Segment:N"), alt.Tooltip("Uplift (pts):Q", format=".2f"),
                         alt.Tooltip("Test CR:Q", format=".2%"), alt.Tooltip("Control CR:Q", format=".2%"),
                         alt.Tooltip("Customers:Q", format=",")])
            zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#172033", strokeDash=[3, 3]).encode(x="x:Q")
            st.altair_chart((err + dot + zero).properties(height=46 * len(seg) + 40).configure_view(strokeWidth=0)
                            .configure_axis(labelColor="#5E6A7D", domainColor="#DCE1E8", gridColor="#E6E9EE"),
                            use_container_width=True)
            st.markdown('<div class="muted">A wide range means too few customers in that segment to be sure. Ranges that '
                        'cross the dashed zero line are not yet distinguishable from no effect.</div>', unsafe_allow_html=True)

        # ---- campaign table
        st.markdown('<div class="sect">Campaigns</div>', unsafe_allow_html=True)
        rows = []
        for k, v in sent.items():
            L, a = v["lift"], v["action"]
            rows.append({"Campaign": k.replace("→", " to "), "Channel": CHANNEL[a.channel],
                         "Offer": "No discount" if a.discount == 0 else f"{a.discount:.0%} off",
                         "Test": L["treated"], "Control": L["holdout"],
                         "CR test": L["cr_test"], "CR control": L["cr_control"], "Uplift (pts)": L["uplift_pp"],
                         "Incr. margin (AED)": L["incremental_gross_margin_aed"],
                         "ROI": L["roi"],
                         "Read": ("Significant" if L["p_value"] < 0.05 else
                                  "Directional" if not v["measurable"] else "Not significant") + (" (p < 0.01)" if L["p_value"] < 0.01 else f" (p = {L['p_value']:.2f})")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, column_config={
            "Campaign": st.column_config.TextColumn(width="medium"),
            "CR test": st.column_config.NumberColumn(format="percent"),
            "CR control": st.column_config.NumberColumn(format="percent"),
            "Uplift (pts)": st.column_config.NumberColumn(format="%.1f"),
            "Incr. margin (AED)": st.column_config.NumberColumn(format="localized"),
            "ROI": st.column_config.NumberColumn(format="%.1f×"),
            "Test": st.column_config.NumberColumn(format="localized"),
            "Control": st.column_config.NumberColumn(format="localized")})
        st.markdown('<div class="muted">Directional: the campaign was too small to judge on its own, so it counts toward '
                    'the overall result above. ROI assumes AED 500 of setup per campaign plus AED 0.01 per email, 0.12 per SMS '
                    'and 0.25 per WhatsApp message. 12-month value is projected from how past cross-store customers spent at that store.'
                    '</div>', unsafe_allow_html=True)

        with st.expander("Model check: did the measurement recover the true effect? (only possible with synthetic data)"):
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
            st.markdown(f"Of the Analyst's {len(opps)} picks, **{found}** are store pairs deliberately built into the data.")

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
