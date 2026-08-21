from agent import Agent
from mas import build_mas_from_specs
from tooling import resolve_tools


class ScriptedAgent(Agent):
    def __init__(self, responses):
        super().__init__(
            system_prompt="Return Final Answer: <number>",
            tools=resolve_tools(["calculate"]),
            require_tool_call=True,
        )
        self.responses = list(responses)

    def _generate_model_n(self, msgs, n, *, tools=None, **gen_kwargs):
        assert n == 1
        return [self.responses.pop(0)]


def test_required_tool_call_is_executed_and_exposed_in_trace():
    agent = ScriptedAgent(
        [
            '<tool_call>{"name":"calculate","arguments":{"expression":"17 * 23"}}</tool_call>',
            "Final Answer: 391",
        ]
    )

    output = agent.generate(agent.format_messages("What is 17 times 23?"))

    assert "TOOL_CALL calculate" in output
    assert '"result": "391"' in output
    assert output.endswith("Final Answer: 391")
    assert agent.last_tool_stats.tool_used_candidates == 1
    assert agent.last_tool_stats.successful_tool_calls == 1
    assert agent.last_tool_stats.tool_names == {"calculate": 1}
    assert agent.last_tool_stats.required_tool_misses == 0


def test_missing_required_call_is_reprompted_before_answering():
    agent = ScriptedAgent(
        [
            "I think the answer is 42.",
            '<tool_call>{"name":"calculate","arguments":{"expression":"6 * 7"}}</tool_call>',
            "Final Answer: 42",
        ]
    )

    agent.generate(agent.format_messages("What is 6 times 7?"))

    assert agent.last_tool_stats.forced_tool_reprompts == 1
    assert agent.last_tool_stats.required_tool_misses == 0
    assert agent.last_tool_stats.tool_calls == 1


def test_disallowed_tool_does_not_satisfy_the_requirement():
    agent = ScriptedAgent(
        [
            '<tool_call>{"name":"verify_arithmetic","arguments":{"expression":"6 * 7","claimed_value":"42"}}</tool_call>',
            "Final Answer: 42",
            '<tool_call>{"name":"calculate","arguments":{"expression":"6 * 7"}}</tool_call>',
            "Final Answer: 42",
        ]
    )

    agent.generate(agent.format_messages("What is 6 times 7?"))

    assert agent.last_tool_stats.tool_calls == 2
    assert agent.last_tool_stats.successful_tool_calls == 1
    assert agent.last_tool_stats.tool_errors == 1
    assert agent.last_tool_stats.tool_names == {"calculate": 1}
    assert agent.last_tool_stats.required_tool_misses == 0


def test_allowed_tool_error_still_records_real_tool_participation():
    agent = ScriptedAgent(
        [
            '<tool_call>{"name":"calculate","arguments":{"expression":"unknown + 1"}}</tool_call>',
            "Final Answer: 1",
        ]
    )

    agent.generate(agent.format_messages("Use the required calculator."))

    assert agent.last_tool_stats.tool_used_candidates == 1
    assert agent.last_tool_stats.tool_names == {"calculate": 1}
    assert agent.last_tool_stats.successful_tool_calls == 0
    assert agent.last_tool_stats.tool_errors == 1
    assert agent.last_tool_stats.required_tool_misses == 0


def test_config_assigns_a_different_tool_to_each_agent():
    mas = build_mas_from_specs(
        model=None,
        tok=None,
        agent_specs=[
            {"name": "Solver", "tools": ["calculate"], "require_tool_call": True},
            {
                "name": "Verifier",
                "tools": ["verify_arithmetic"],
                "require_tool_call": True,
            },
        ],
        edges=[[-1, 0], [-1, 1], [0, 1]],
    )

    assert [tool.name for tool in mas.agents[0].tools] == ["calculate"]
    assert [tool.name for tool in mas.agents[1].tools] == ["verify_arithmetic"]
    assert all(agent.require_tool_call for agent in mas.agents)
