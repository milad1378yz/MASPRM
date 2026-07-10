import math
from types import SimpleNamespace

import pytest
import torch

from experiments.core import (
    HandoffDecodeResult,
    TokenStats,
    compute_batch_logprob_and_token_counts,
    compute_logprob_and_token_counts,
    handoff_sbs_decode,
    render_handoff_state_text,
)
from handoff_mas import FINAL_RECIPIENT, HandoffMAS, HandoffRun, RoutedTurn


class _CharTokenizer:
    """Small reversible tokenizer suitable for accounting and truncation tests."""

    def __call__(
        self,
        text,
        *,
        return_tensors=None,
        add_special_tokens=False,
        **_kwargs,
    ):
        del add_special_tokens
        ids = [ord(char) for char in str(text)]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, *, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(int(token_id)) for token_id in ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt=True,
    ):
        rendered = "\n".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "\nassistant:"
        if tokenize:
            return [ord(char) for char in rendered]
        return rendered


class _ScriptedAgent:
    def __init__(self, name, tokenizer, batches):
        self.name = name
        self.tok = tokenizer
        self.system_prompt = f"Act as {name}."
        self._batches = list(batches)
        self.calls = []

    def format_messages(self, user_content):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def generate_n(self, messages, n, **gen_kwargs):
        self.calls.append((messages, n, gen_kwargs))
        outputs = self._batches.pop(0)
        assert len(outputs) == n
        return outputs


class _BatchScorer:
    def __init__(self):
        self.calls = []

    def __call__(self, _text):
        raise AssertionError("The decoder should use the available batch scorer.")

    def batch(self, texts):
        self.calls.append(list(texts))
        scores = []
        for text in texts:
            latest_turn = text.rsplit("SPEAKER:", 1)[-1]
            scores.append(1.0 if "SELECT_ME" in latest_turn else -1.0)
        return scores


class _LogprobTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, _messages, *, tokenize, add_generation_prompt=True):
        del add_generation_prompt
        return [1, 2] if tokenize else "prompt"

    def __call__(self, text, *, return_tensors=None, add_special_tokens=False):
        del add_special_tokens
        ids = [{"a": 3, "b": 4, "c": 5}[char] for char in str(text)]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}


class _LogprobModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(
        self,
        input_ids,
        attention_mask=None,
        use_cache=None,
        logits_to_keep=0,
    ):
        del attention_mask, use_cache
        batch, length = input_ids.shape
        vocab = torch.arange(8, device=input_ids.device, dtype=torch.float32)
        positions = torch.arange(length, device=input_ids.device, dtype=torch.float32)
        logits = (
            positions.view(1, length, 1) * vocab.view(1, 1, 8) * 0.05
            + vocab.view(1, 1, 8) * 0.2
        )
        logits = logits.expand(batch, -1, -1) + self.anchor
        if isinstance(logits_to_keep, torch.Tensor):
            logits = logits[:, logits_to_keep, :]
        return SimpleNamespace(logits=logits)


def test_batched_policy_logprob_matches_scalar_scoring():
    agent = SimpleNamespace(tok=_LogprobTokenizer(), model=_LogprobModel())
    messages = [{"role": "user", "content": "question"}]
    completions = ["a", "bc"]

    batch_scores, prompt_len, lengths = compute_batch_logprob_and_token_counts(
        agent,
        messages,
        completions,
    )
    scalar = [
        compute_logprob_and_token_counts(agent, messages, completion)[0]
        for completion in completions
    ]

    assert prompt_len == 2
    assert lengths == [1, 2]
    assert batch_scores == pytest.approx(scalar)


def _render_action(speaker, recipient, message, answer):
    return (
        f"SPEAKER: {speaker}\n"
        f"RECIPIENT: {recipient}\n"
        f"MESSAGE: {message}\n"
        f"CURRENT_ANSWER: {answer}"
    )


def _routed_turn(speaker, recipient, message, answer):
    return RoutedTurn(
        speaker=speaker,
        recipient=recipient,
        message=message,
        current_answer=answer,
    )


def test_handoff_prm_decoder_batches_proposals_and_selects_best_action():
    tokenizer = _CharTokenizer()
    agent_a = _ScriptedAgent(
        "A",
        tokenizer,
        [
            [
                _render_action("A", "B", "ordinary", "1"),
                _render_action("A", "B", "SELECT_ME first", "2"),
            ]
        ],
    )
    agent_b = _ScriptedAgent(
        "B",
        tokenizer,
        [
            [
                _render_action("B", FINAL_RECIPIENT, "ordinary", "3"),
                _render_action("B", FINAL_RECIPIENT, "SELECT_ME final", "4"),
            ]
        ],
    )
    scorer = _BatchScorer()
    mas = HandoffMAS([agent_a, agent_b], ["A", "B"])

    result = handoff_sbs_decode(
        mas,
        "question",
        route_mode="agent",
        score_type="prm",
        score_fn=scorer,
        prm_tokenizer=tokenizer,
        n_candidates=2,
        min_turns=2,
        max_turns=2,
        initial_speaker="A",
        max_context_tokens=10_000,
    )

    assert result.answer == "4"
    assert [turn.message for turn in result.run.turns] == [
        "SELECT_ME first",
        "SELECT_ME final",
    ]
    assert result.run.stop_reason == "final"
    assert result.proposals == 4
    assert result.prm_evaluations == 4
    assert result.usage.agent_runs == 2
    assert result.usage.prm_calls == 2
    assert len(agent_a.calls) == len(agent_b.calls) == 1
    assert agent_a.calls[0][1] == agent_b.calls[0][1] == 2
    assert [len(call) for call in scorer.calls] == [2, 2]
    assert "SELECT_ME first" in scorer.calls[1][0]


def test_render_handoff_state_drops_oldest_whole_turns_and_keeps_newest():
    tokenizer = _CharTokenizer()
    question = "QUESTION|"
    turns = [
        _routed_turn("A", "B", "OLDEST_MARKER" * 4, "1"),
        _routed_turn("B", "C", "MIDDLE_MARKER" * 4, "2"),
        _routed_turn("C", FINAL_RECIPIENT, "NEWEST_MARKER", "3"),
    ]
    newest_piece = turns[-1].render() + "</step>"
    limit = len(question + newest_piece)

    rendered = render_handoff_state_text(
        question,
        turns,
        tokenizer=tokenizer,
        max_context_tokens=limit,
    )

    assert rendered.truncated is True
    assert rendered.text == question + newest_piece
    assert rendered.text.startswith(question)
    assert rendered.text.endswith(newest_piece)
    assert "OLDEST_MARKER" not in rendered.text
    assert "MIDDLE_MARKER" not in rendered.text
    assert rendered.token_count == limit
    assert rendered.original_token_count > rendered.token_count


def test_handoff_metrics_are_deterministic_for_known_route():
    turns = (
        _routed_turn("A", "B", "one", "1"),
        _routed_turn("B", "A", "two", "2"),
        _routed_turn("A", "B", "three", "3"),
        _routed_turn("B", FINAL_RECIPIENT, "four", "4"),
    )
    result = HandoffDecodeResult(
        run=HandoffRun(question="q", turns=turns, stop_reason="final"),
        usage=TokenStats(prompt=11, generated=12, scorer=13, prm_calls=2, agent_runs=4),
        n_roles=3,
        proposals=8,
        parse_failures=2,
        prm_evaluations=6,
        prm_context_tokens=[10, 20, 30],
        prm_original_context_tokens=[10, 25, 40],
        prm_truncations=1,
    )

    metrics = result.metrics(thresholds=(1, 3, 5))
    expected_entropy = -(
        (2 / 3) * math.log2(2 / 3) + (1 / 3) * math.log2(1 / 3)
    )

    assert metrics["answer"] == "4"
    assert metrics["prefix_answers"] == {"1": "1", "3": "3", "5": "4"}
    assert metrics["prefix_depths"] == {"1": 1, "3": 3, "5": 4}
    assert metrics["active_roles"] == ["A", "B"]
    assert metrics["unique_agents"] == 2
    assert metrics["directed_role_edges"] == ["A -> B", "B -> A", "A -> B"]
    assert metrics["unique_directed_role_edges"] == 2
    assert metrics["role_revisits"] == 2
    assert metrics["role_revisit_rate"] == pytest.approx(0.5)
    assert metrics["route_entropy_bits"] == pytest.approx(expected_entropy)
    assert metrics["route_entropy_normalized"] == pytest.approx(
        expected_entropy / math.log2(6)
    )
    assert metrics["generated_tokens"] == 12
    assert metrics["agent_calls"] == 4
    assert metrics["parse_failure_rate"] == pytest.approx(0.25)
    assert metrics["prm_context_token_median"] == 20.0
    assert metrics["prm_context_token_p95"] == 30.0
    assert metrics["prm_truncation_rate"] == pytest.approx(1 / 3)
