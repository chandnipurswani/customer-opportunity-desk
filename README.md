# Customer Opportunity Desk

**An agentic decision-support prototype for customer growth teams.** Every week, AI agents find which shoppers of one store are likely to start buying at another store in the same retail group, design how to reach them, and, once a manager approves, measure whether it worked against a control group.

**[Live demo](https://customer-opportunity-desk.streamlit.app/)** · **[Concept note](https://claude.ai/artifact/DEZTMMYhQquuuJ8VY1Vr5U)**

> All data is synthetic. Al Waha Retail and its six stores (Fashion, Home, Sports, Beauty, Kids, Electronics) are fictional.

---

## The problem

Multi-brand retail groups in the GCC run many stores under one loyalty programme, yet most customers shop at just one. Each store team optimises its own customers, so nobody owns moving a shopper from one store to the next. Spotting the right store pairs is slow manual analysis, and cross-store campaigns are rarely measured properly, often with discounts that give the margin away.

## What it does

| Step | Who | What happens |
|---|---|---|
| 1. Find | **Analyst agent** (Claude) | Ranks store pairs by how much more likely one store's shoppers are to buy at the other, and explains the evidence |
| 2. Design | **Strategist agent** (Claude) | Compares channels and offers, message first and a discount only if it earns more margin, then drafts the customer message in English and Arabic |
| 3. Check | **Engine** (code) | Every number, consent and contact-frequency rules, discount ceiling, budget, control-group size |
| 4. Decide | **The manager** | Approves, edits or rejects in an inbox. Nothing is sent without approval |
| 5. Measure | **Engine** (code) | Simulated send; test vs control conversion, incremental gross margin, ROI, projected 12-month customer value |

**It is not a chatbot.** It moves from information (what changed) to decision support (what to do, and why) to action (an approved, executed and measured campaign).

## Product decisions worth noting

- **Agents reason and write; code calculates.** Agents call engine tools for every figure, and an automatic check flags any number in their output that didn't come from a tool. In the first live run it caught the Strategist doing its own subtraction. The fix was a new `compare_options` tool, not a prompt tweak.
- **Incremental gross margin is the north star**, not revenue, so the system can't buy results with discounts.
- **Control groups are sized by a power calculation.** In simulation a fixed 10% control missed real lifts up to half the time. Most single cross-store campaigns are too small to read alone, so a permanent **10% global holdout** measures the programme as a whole.
- **Human approval for every action in v0.** More autonomy for low-risk actions is a roadmap step, earned through evaluation results.
- **Synthetic data with known answers.** Planted store affinities and true campaign effects let the evaluation test whether the agents find the signal and whether the measurement recovers the truth.

The full reasoning, evaluation framework, business case and roadmap are in the [concept note](https://claude.ai/artifact/DEZTMMYhQquuuJ8VY1Vr5U).

## First results

- The Analyst's top 3 were the **3 store affinities planted in the data**, found without access to the answer key.
- **Zero unsupported numbers** in agent output after the compare tool was added.
- Measured ranges contained the true planted effect in about **9 of 10** simulated campaigns.
- A weekly agent run takes **about 1 minute and roughly $0.20**.

## Run it locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                  # 13 tests, no API key needed
export ANTHROPIC_API_KEY=...               # for live agent runs
python -m desk.agents                      # weekly agent run, saved to runs/latest.json
streamlit run app/streamlit_app.py         # the desk
```

`DESK_MOCK=1 python -m desk.agents` runs an offline stand-in with no API calls.

## Project layout

```
desk/synthetic.py   fictional retail group: 50k customers, purchases, consent, planted affinities
desk/engine.py      deterministic engine: affinity, margin estimate, policy rules, control groups, measurement
desk/agents.py      Analyst and Strategist agents (Claude tool use), number check, saved runs
app/                Streamlit app: weekly inbox and performance dashboard
tests/              engine, agent loop and number-check tests
```

## Assumptions to replace with real data

Messaging costs (AED 0.01 per email, 0.12 per SMS, 0.25 per WhatsApp message), AED 500 setup per campaign, and the planning-estimate uplift parameters are working assumptions. Only the test-vs-control comparison counts as a result.

---

Built by [Chandni Purswani](https://www.linkedin.com/in/chandni-purswani/) · Not affiliated with any real retailer.
