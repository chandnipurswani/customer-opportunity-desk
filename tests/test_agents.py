import json
from types import SimpleNamespace as NS

import pytest

from desk import agents as A
from desk.engine import Engine
from desk.synthetic import generate


@pytest.fixture(scope="module")
def eng():
    return Engine(generate(n_customers=20_000, seed=7))


class FakeClient:
    """Scripted stand-in for anthropic.Anthropic(): first asks for a tool, then answers in JSON."""
    def __init__(self, final: dict):
        self.final, self.calls = final, 0
        self.messages = self

    def create(self, **kw):
        self.calls += 1
        usage = NS(input_tokens=100, output_tokens=50)
        if self.calls == 1:
            return NS(stop_reason="tool_use", usage=usage, content=[
                NS(type="tool_use", id="t1", name="pair_detail", input={"from_brand": "Mirage", "to_brand": "Glow"})])
        return NS(stop_reason="end_turn", usage=usage, content=[NS(type="text", text="Here:\n" + json.dumps(self.final))])


def test_tool_loop_runs_engine_tools_and_parses_json(eng):
    final = {"opportunities": [{"rank": 1, "from_brand": "Mirage", "to_brand": "Glow", "headline": "x"}]}
    r = A.run_agent(eng, "sys", "go", ["pair_detail"], client=FakeClient(final))
    assert r["output"] == final
    assert r["trace"][0]["tool"] == "pair_detail" and r["trace"][0]["output"]["to_brand"] == "Glow"
    assert r["usage"] == {"input_tokens": 200, "output_tokens": 100}


def test_faithfulness_flags_invented_numbers(eng):
    d = eng.pair_detail("Mirage", "Glow")
    ok = {"output": {"text": f"Audience of {d['audience_size']} customers"}, "trace": [{"output": d}]}
    bad = {"output": {"text": "Audience of 123456 customers"}, "trace": [{"output": d}]}
    assert A.faithfulness(ok)["faithful"]
    assert A.faithfulness(bad)["unsupported"] == [123456.0]


def test_mock_cycle_respects_policy_and_top_n(eng):
    c = A.run_cycle(eng, mock=True)
    assert len(c["strategist"]) == A.TOP_N
    for s in c["strategist"]:
        o = s["output"]
        assert eng.check_policy(o["from_brand"], o["to_brand"], o["channel"], o["discount"])["passed"]
