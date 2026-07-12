import operator
import re
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent import Agent

FINAL_RECIPIENT = "FINAL"
_RUN_CONFIG_KEY = "handoff_run"
_FIELD_RE = re.compile(r"(?im)^\s*(SPEAKER|RECIPIENT|MESSAGE|CURRENT_ANSWER)\s*:\s*")


def _role_key(value: str) -> str:
    return " ".join(str(value).strip().split()).casefold()


@dataclass(frozen=True)
class RoutedTurn:
    speaker: str
    recipient: str
    message: str
    current_answer: str
    raw: str = ""

    def render(self) -> str:
        return (
            f"SPEAKER: {self.speaker}\n"
            f"RECIPIENT: {self.recipient}\n"
            f"MESSAGE: {self.message}\n"
            f"CURRENT_ANSWER: {self.current_answer}"
        )


@dataclass(frozen=True)
class HandoffContext:
    question: str
    turns: Tuple[RoutedTurn, ...]
    visible_turns: Tuple[RoutedTurn, ...]
    speaker_idx: int
    speaker: str
    min_turns: int
    max_turns: int

    @property
    def turn_number(self) -> int:
        return len(self.turns) + 1

    @property
    def final_allowed(self) -> bool:
        return self.turn_number >= self.min_turns


@dataclass(frozen=True)
class HandoffRun:
    question: str
    turns: Tuple[RoutedTurn, ...]
    stop_reason: str

    @property
    def answer(self) -> str:
        return self.turns[-1].current_answer if self.turns else ""

    @property
    def depth(self) -> int:
        return len(self.turns)

    def prefix_answers(self, thresholds: Sequence[int] = (4, 8, 12)) -> Dict[int, str]:
        """Return the latest answer at each budget, carrying early stops forward."""
        if not self.turns:
            return {int(t): "" for t in thresholds}
        return {
            int(t): self.turns[min(max(1, int(t)), len(self.turns)) - 1].current_answer
            for t in thresholds
        }


ActionSelector = Callable[[HandoffContext], RoutedTurn]


@dataclass(frozen=True)
class _RunParams:
    """Run-scoped dependencies passed through LangGraph's `configurable`."""

    selector: ActionSelector
    min_turns: int
    max_turns: int
    allow_self: bool


class _HandoffState(TypedDict):
    question: str
    turns: Annotated[List[RoutedTurn], operator.add]
    next_speaker: int
    stop_reason: str


def parse_routed_turn(
    text: str,
    *,
    expected_speaker: str,
    role_names: Sequence[str],
    allowed_recipients: Optional[Sequence[str]] = None,
    allow_final: bool = False,
) -> RoutedTurn:
    """Parse and strictly validate the four-field routed-action contract."""
    raw = str(text or "").strip()
    matches = list(_FIELD_RE.finditer(raw))
    expected_fields = ["SPEAKER", "RECIPIENT", "MESSAGE", "CURRENT_ANSWER"]
    found_fields = [match.group(1).upper() for match in matches]
    has_leading_text = bool(matches and raw[: matches[0].start()].strip())
    if has_leading_text or found_fields != expected_fields:
        raise ValueError(
            "Routed action must contain SPEAKER, RECIPIENT, MESSAGE, and "
            "CURRENT_ANSWER exactly once and in that order."
        )

    values: Dict[str, str] = {}
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        values[match.group(1).upper()] = raw[match.end() : end].strip()

    canonical_roles = {_role_key(name): str(name) for name in role_names}
    speaker_key = _role_key(values["SPEAKER"])
    expected_key = _role_key(expected_speaker)
    if speaker_key != expected_key:
        raise ValueError(
            f"Action speaker {values['SPEAKER']!r} does not match active role "
            f"{expected_speaker!r}."
        )

    recipient_value = values["RECIPIENT"]
    if _role_key(recipient_value) == _role_key(FINAL_RECIPIENT):
        if not allow_final:
            raise ValueError("FINAL is not allowed before the minimum turn count.")
        recipient = FINAL_RECIPIENT
    else:
        recipient_key = _role_key(recipient_value)
        if recipient_key not in canonical_roles:
            raise ValueError(f"Unknown recipient role: {recipient_value!r}.")
        recipient = canonical_roles[recipient_key]

    if allowed_recipients is not None:
        allowed = {_role_key(name) for name in allowed_recipients}
        if _role_key(recipient) not in allowed:
            raise ValueError(f"Recipient {recipient!r} is not permitted for this turn.")

    message = values["MESSAGE"].strip()
    current_answer = values["CURRENT_ANSWER"].strip()
    if not message:
        raise ValueError("MESSAGE cannot be empty.")
    if not current_answer:
        raise ValueError("CURRENT_ANSWER cannot be empty.")

    return RoutedTurn(
        speaker=canonical_roles.get(speaker_key, expected_speaker),
        recipient=recipient,
        message=message,
        current_answer=current_answer,
        raw=raw,
    )


class HandoffMAS:
    """A compiled cyclic LangGraph for private, repeated role handoffs.

    Candidate generation and selection happen inside one role-node call. The
    graph state is the ordered transcript plus routing data; each node appends
    exactly the selected action through a reducer and hands control directly
    to the chosen recipient. Private per-role views are derived from the
    transcript, so there is no second copy of history to keep consistent.
    """

    def __init__(self, agents: Sequence[Agent], role_names: Sequence[str]):
        if len(agents) != len(role_names) or not agents:
            raise ValueError(
                "agents and role_names must have the same non-zero length."
            )

        self.agents = list(agents)
        self.role_names = [str(name).strip() for name in role_names]
        if any(not name for name in self.role_names):
            raise ValueError("Every handoff role must have a non-empty name.")

        self._role_indices = {
            _role_key(name): idx for idx, name in enumerate(self.role_names)
        }
        if len(self._role_indices) != len(self.role_names):
            raise ValueError("Handoff role names must be unique (case-insensitive).")
        if _role_key(FINAL_RECIPIENT) in self._role_indices:
            raise ValueError(
                f"{FINAL_RECIPIENT!r} is reserved for termination and cannot "
                "name a role."
            )

        self.n = len(self.agents)
        self._node_names = [f"handoff_role_{idx}" for idx in range(self.n)]
        self._graph = self._build_graph()

    def role_index(self, role: Union[int, str]) -> int:
        if isinstance(role, int):
            if 0 <= role < self.n:
                return role
            raise ValueError(f"Role index {role} is outside [0, {self.n}).")
        key = _role_key(role)
        if key not in self._role_indices:
            raise ValueError(f"Unknown handoff role: {role!r}.")
        return self._role_indices[key]

    def visible_turns(
        self, role: str, turns: Sequence[RoutedTurn]
    ) -> Tuple[RoutedTurn, ...]:
        """A role's private view: every turn it spoke or received, in order."""
        key = _role_key(role)
        return tuple(
            turn
            for turn in turns
            if _role_key(turn.speaker) == key or _role_key(turn.recipient) == key
        )

    def allowed_recipients(
        self,
        context: HandoffContext,
        *,
        allow_self: bool = False,
        include_final: Optional[bool] = None,
    ) -> List[str]:
        recipients = [
            role
            for idx, role in enumerate(self.role_names)
            if allow_self or idx != context.speaker_idx
        ]
        if context.final_allowed if include_final is None else include_final:
            recipients.append(FINAL_RECIPIENT)
        return recipients

    def agent_user_content(
        self,
        context: HandoffContext,
        *,
        permitted_recipients: Sequence[str],
    ) -> str:
        history = "\n\n".join(turn.render() for turn in context.visible_turns)
        permitted = ", ".join(permitted_recipients)
        final_rule = (
            "FINAL IS FORBIDDEN ON THIS TURN."
            if FINAL_RECIPIENT not in permitted_recipients
            else "FINAL may be used only when the answer is fully solved and verified."
        )
        parts = [
            f"ORIGINAL QUESTION:\n{context.question}",
            (
                "PRIVATE ROUTED HISTORY:\n" + history
                if history
                else "PRIVATE ROUTED HISTORY:\n(none)"
            ),
            f"ACTIVE SPEAKER: {context.speaker}",
            f"PERMITTED RECIPIENTS: {permitted}",
            (
                "Produce exactly one action using the required four fields. "
                "Do not address any recipient outside the permitted list."
            ),
            (
                "STRICT OUTPUT CHECK:\n"
                f"{final_rule}\n"
                "Do not output ACTION or any prose before SPEAKER. Fill this exact template:\n"
                f"SPEAKER: {context.speaker}\n"
                f"RECIPIENT: <exactly one of: {permitted}>\n"
                "MESSAGE: <your reasoning or advice>\n"
                "CURRENT_ANSWER: <the best answer so far>"
            ),
        ]
        return "\n\n".join(parts)

    def agent_messages(
        self,
        context: HandoffContext,
        *,
        permitted_recipients: Sequence[str],
    ):
        return self.agents[context.speaker_idx].format_messages(
            self.agent_user_content(
                context,
                permitted_recipients=permitted_recipients,
            )
        )

    def sample_candidate_texts(
        self,
        context: HandoffContext,
        *,
        n_candidates: int,
        permitted_recipients: Sequence[str],
        **gen_kwargs: Any,
    ) -> Tuple[List[str], Any]:
        """Batch all proposals for a turn in one policy-model call."""
        if n_candidates < 1:
            raise ValueError("n_candidates must be at least 1.")
        messages = self.agent_messages(
            context,
            permitted_recipients=permitted_recipients,
        )
        outputs = self.agents[context.speaker_idx].generate_n(
            messages,
            n=n_candidates,
            **gen_kwargs,
        )
        return [str(output).strip() for output in outputs], messages

    def run(
        self,
        question: str,
        selector: ActionSelector,
        *,
        initial_speaker: Union[int, str] = 0,
        min_turns: int = 4,
        max_turns: int = 12,
        allow_self: bool = False,
    ) -> HandoffRun:
        min_turns = int(min_turns)
        max_turns = int(max_turns)
        if min_turns < 1:
            raise ValueError("min_turns must be at least 1.")
        if max_turns < min_turns:
            raise ValueError("max_turns must be greater than or equal to min_turns.")

        initial_state: _HandoffState = {
            "question": str(question),
            "turns": [],
            "next_speaker": self.role_index(initial_speaker),
            "stop_reason": "",
        }
        params = _RunParams(
            selector=selector,
            min_turns=min_turns,
            max_turns=max_turns,
            allow_self=bool(allow_self),
        )
        state = self._graph.invoke(
            initial_state,
            config={
                "recursion_limit": max_turns + 2,
                "configurable": {_RUN_CONFIG_KEY: params},
            },
        )
        return HandoffRun(
            question=state["question"],
            turns=tuple(state["turns"]),
            stop_reason=state["stop_reason"] or "max_turns",
        )

    def _entry_speaker(self, state: _HandoffState) -> str:
        return self._node_names[state["next_speaker"]]

    def _role_node(self, speaker_idx: int):
        speaker = self.role_names[speaker_idx]

        def run(state: _HandoffState, config: RunnableConfig) -> Command:
            params: _RunParams = config["configurable"][_RUN_CONFIG_KEY]
            turns = tuple(state["turns"])
            context = HandoffContext(
                question=state["question"],
                turns=turns,
                visible_turns=self.visible_turns(speaker, turns),
                speaker_idx=speaker_idx,
                speaker=speaker,
                min_turns=params.min_turns,
                max_turns=params.max_turns,
            )
            turn = params.selector(context)
            if _role_key(turn.speaker) != _role_key(speaker):
                raise ValueError(
                    f"Selected action speaker {turn.speaker!r} does not match "
                    f"{speaker!r}."
                )

            is_final = _role_key(turn.recipient) == _role_key(FINAL_RECIPIENT)
            if is_final and not context.final_allowed:
                raise ValueError("FINAL is not allowed before the minimum turn count.")

            next_speaker = speaker_idx
            if not is_final:
                next_speaker = self.role_index(turn.recipient)
                if not params.allow_self and next_speaker == speaker_idx:
                    raise ValueError("Self handoffs are disabled.")

            done = is_final or context.turn_number >= params.max_turns
            update: Dict[str, Any] = {
                "turns": [turn],
                "next_speaker": next_speaker,
            }
            if done:
                update["stop_reason"] = "final" if is_final else "max_turns"
            return Command(
                update=update,
                goto=END if done else self._node_names[next_speaker],
            )

        return run

    def _build_graph(self):
        graph = StateGraph(_HandoffState)
        destinations = tuple([*self._node_names, END])
        for idx, node_name in enumerate(self._node_names):
            graph.add_node(
                node_name,
                self._role_node(idx),
                destinations=destinations,
            )
        graph.add_conditional_edges(START, self._entry_speaker, self._node_names)
        return graph.compile(name="handoff_mas")


def build_handoff_mas_from_specs(
    model: Any,
    tok: Any,
    agent_specs: Sequence[Dict[str, Any]],
) -> HandoffMAS:
    role_names = [str(spec.get("name", "")).strip() for spec in agent_specs]
    agents = [
        Agent(
            model,
            tok,
            name=role_name,
            system_prompt=str(spec.get("system_prompt", "")),
            max_new_tokens=int(spec.get("max_new_tokens", 512)),
        )
        for spec, role_name in zip(agent_specs, role_names)
    ]
    return HandoffMAS(agents, role_names)
