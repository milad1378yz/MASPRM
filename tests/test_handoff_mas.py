import pytest

from agent import Agent
from handoff_mas import (
    FINAL_RECIPIENT,
    HandoffMAS,
    RoutedTurn,
    build_handoff_mas_from_specs,
    parse_routed_turn,
)


class _InertAgent:
    """The graph selector supplies actions, so these agents never generate."""


def _turn(speaker: str, recipient: str, message: str, answer: str) -> RoutedTurn:
    return RoutedTurn(
        speaker=speaker,
        recipient=recipient,
        message=message,
        current_answer=answer,
    )


def test_parse_routed_turn_canonicalizes_roles_and_keeps_multiline_message():
    text = """\
 speaker: problem   analyst
RECIPIENT: critic
MESSAGE: Check the setup first.
Then verify the arithmetic.
current_answer: 42
"""

    turn = parse_routed_turn(
        text,
        expected_speaker="Problem Analyst",
        role_names=["Problem Analyst", "Critic"],
        allow_final=False,
    )

    assert turn.speaker == "Problem Analyst"
    assert turn.recipient == "Critic"
    assert turn.message == "Check the setup first.\nThen verify the arithmetic."
    assert turn.current_answer == "42"
    assert turn.raw == text.strip()


@pytest.mark.parametrize(
    "text",
    [
        # Missing CURRENT_ANSWER.
        "SPEAKER: A\nRECIPIENT: B\nMESSAGE: work",
        # Duplicate fields are not permitted.
        (
            "SPEAKER: A\nRECIPIENT: B\nMESSAGE: work\n"
            "MESSAGE: more\nCURRENT_ANSWER: 1"
        ),
        # Fields must occur in the declared order.
        "RECIPIENT: B\nSPEAKER: A\nMESSAGE: work\nCURRENT_ANSWER: 1",
        # Both free-text values are required.
        "SPEAKER: A\nRECIPIENT: B\nMESSAGE:\nCURRENT_ANSWER: 1",
        "SPEAKER: A\nRECIPIENT: B\nMESSAGE: work\nCURRENT_ANSWER:",
    ],
)
def test_parse_routed_turn_rejects_malformed_contract(text):
    with pytest.raises(ValueError):
        parse_routed_turn(
            text,
            expected_speaker="A",
            role_names=["A", "B"],
        )


def test_parse_routed_turn_rejects_wrong_speaker_recipient_and_early_final():
    with pytest.raises(ValueError, match="does not match active role"):
        parse_routed_turn(
            "SPEAKER: B\nRECIPIENT: A\nMESSAGE: work\nCURRENT_ANSWER: 1",
            expected_speaker="A",
            role_names=["A", "B"],
        )

    with pytest.raises(ValueError, match="Unknown recipient"):
        parse_routed_turn(
            "SPEAKER: A\nRECIPIENT: C\nMESSAGE: work\nCURRENT_ANSWER: 1",
            expected_speaker="A",
            role_names=["A", "B"],
        )

    with pytest.raises(ValueError, match="minimum turn"):
        parse_routed_turn(
            "SPEAKER: A\nRECIPIENT: FINAL\nMESSAGE: done\nCURRENT_ANSWER: 1",
            expected_speaker="A",
            role_names=["A", "B"],
            allow_final=False,
        )


def test_parse_routed_turn_rejects_text_before_first_field():
    text = (
        "Here is my routed action.\n"
        "SPEAKER: A\nRECIPIENT: B\nMESSAGE: work\nCURRENT_ANSWER: 1"
    )

    with pytest.raises(ValueError):
        parse_routed_turn(
            text,
            expected_speaker="A",
            role_names=["A", "B"],
        )


def test_compiled_graph_supports_repeated_roles_and_private_histories():
    mas = HandoffMAS([_InertAgent(), _InertAgent(), _InertAgent()], ["A", "B", "C"])
    scripted = [
        _turn("A", "B", "A to B", "1"),
        _turn("B", "C", "B to C", "2"),
        _turn("C", "B", "C back to B", "3"),
        _turn("B", FINAL_RECIPIENT, "B finishes", "4"),
    ]
    contexts = []

    def select(context):
        contexts.append(context)
        return scripted[len(contexts) - 1]

    run = mas.run(
        "private question",
        select,
        initial_speaker="A",
        min_turns=4,
        max_turns=8,
    )

    assert run.stop_reason == "final"
    assert run.depth == 4
    assert run.answer == "4"
    assert [turn.speaker for turn in run.turns] == ["A", "B", "C", "B"]

    # C receives B's message, but not the earlier A -> B exchange.
    assert [(turn.speaker, turn.recipient) for turn in contexts[2].visible_turns] == [
        ("B", "C")
    ]
    assert all(turn.message != "A to B" for turn in contexts[2].visible_turns)

    # On B's second activation it sees its inbound messages and its own output.
    assert [turn.message for turn in contexts[3].visible_turns] == [
        "A to B",
        "B to C",
        "C back to B",
    ]
    assert all(context.question == "private question" for context in contexts)


def test_compiled_graph_rejects_final_before_minimum_turn():
    mas = HandoffMAS([_InertAgent(), _InertAgent()], ["A", "B"])

    def select(context):
        return _turn(context.speaker, FINAL_RECIPIENT, "too soon", "1")

    with pytest.raises(ValueError, match="minimum turn"):
        mas.run(
            "question",
            select,
            initial_speaker="A",
            min_turns=2,
            max_turns=4,
        )


def test_compiled_graph_stops_at_maximum_turn_limit():
    mas = HandoffMAS([_InertAgent(), _InertAgent()], ["A", "B"])
    turn_numbers = []

    def select(context):
        turn_numbers.append(context.turn_number)
        recipient = "B" if context.speaker == "A" else "A"
        return _turn(context.speaker, recipient, "continue", str(context.turn_number))

    run = mas.run(
        "question",
        select,
        initial_speaker="A",
        min_turns=2,
        max_turns=3,
    )

    assert run.stop_reason == "max_turns"
    assert run.depth == 3
    assert run.answer == "3"
    assert turn_numbers == [1, 2, 3]
    assert [turn.speaker for turn in run.turns] == ["A", "B", "A"]


def test_agent_generation_cap_and_handoff_builder_preserve_role_names():
    finalizer = Agent(name="Finalizer", max_new_tokens=256)

    assert finalizer._generation_kwargs(max_new_tokens=1024)["max_new_tokens"] == 256
    assert finalizer._generation_kwargs(max_new_tokens=128)["max_new_tokens"] == 128
    assert finalizer._generation_kwargs()["max_new_tokens"] == 256

    specs = [
        {"name": "Problem Analyst", "system_prompt": "Analyze.", "max_new_tokens": 512},
        {"name": "Finalizer", "system_prompt": "Finish.", "max_new_tokens": 256},
    ]
    mas = build_handoff_mas_from_specs(model=None, tok=object(), agent_specs=specs)

    assert mas.role_names == ["Problem Analyst", "Finalizer"]
    assert [agent.name for agent in mas.agents] == ["Problem Analyst", "Finalizer"]
    assert [agent.max_new_tokens for agent in mas.agents] == [512, 256]
