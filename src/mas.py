from typing import Annotated, Dict, List, Optional, Set, Tuple, Any, TypedDict

from agent import Agent
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


def _merge_nested_dicts(
    left: Optional[Dict[int, Dict[int, str]]], right: Optional[Dict[int, Dict[int, str]]]
) -> Dict[int, Dict[int, str]]:
    merged: Dict[int, Dict[int, str]] = {
        int(k): dict(v) for k, v in (left or {}).items()
    }
    for key, value in (right or {}).items():
        key = int(key)
        child_updates = dict(value)
        if key in merged:
            next_value = dict(merged[key])
            next_value.update(child_updates)
            merged[key] = next_value
        else:
            merged[key] = child_updates
    return merged


def _merge_dicts(left: Optional[Dict[int, Any]], right: Optional[Dict[int, Any]]) -> Dict[int, Any]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class _MASState(TypedDict):
    query: str
    inbox: Annotated[Dict[int, Dict[int, str]], _merge_nested_dicts]
    agent_io: Annotated[Dict[int, Dict[str, Dict[int, str]]], _merge_dicts]
    primary_out: Annotated[Dict[int, str], _merge_dicts]
    executed: Annotated[Dict[int, bool], _merge_dicts]
    active_agents: Optional[Set[int]]
    forced_outputs: Dict[int, List[str]]
    rounds: int
    max_rounds: int
    execution_order: List[int]


class MAS:
    INPUT = -1  # special node id for the external query

    def __init__(self, edges: List[Tuple[int, int]], agnets: List[Agent]):
        self.edges = [(int(s), int(t)) for s, t in edges]
        self.agents = agnets
        self.n = len(agnets)

        self.children: Dict[int, List[int]] = {}
        self.parents: Dict[int, List[int]] = {}
        for s, t in self.edges:
            self.children.setdefault(s, []).append(t)
            self.parents.setdefault(t, []).append(s)
        for node in self.children:
            self.children[node].sort()
        for node in self.parents:
            self.parents[node].sort()

        self.edge_index = {e: i for i, e in enumerate(self.edges)}
        self.sinks = [i for i in range(self.n) if len(self.children.get(i, [])) == 0]
        self.required_parents: Dict[int, Set[int]] = {
            i: set(self.parents.get(i, [])) for i in range(self.n)
        }
        self.parent_order: Dict[int, List[int]] = {
            i: sorted(self.required_parents[i]) for i in range(self.n)
        }
        self._graph = self._build_graph()

    # It returns a list of output strings (one per outgoing edge, or a single value to broadcast).
    def step(self, messages: List[str], idx: int, **gen_kwargs) -> List[str]:
        sys_prompt = getattr(self.agents[idx], "system_prompt", "You are a helpful assistant.")
        content = "\n\n".join(messages).strip()
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content},
        ]
        out = self.agents[idx].generate(msgs, **gen_kwargs)
        # Normalize to List[str]
        if isinstance(out, list):
            return [str(x) for x in out]
        return [str(out)]

    def _agent_node_name(self, idx: int) -> str:
        return f"agent_{idx}"

    def _seed_inbox_with_query(self, query: str) -> Dict[int, Dict[int, str]]:
        inbox: Dict[int, Dict[int, str]] = {}
        for agent_idx in self.children.get(self.INPUT, []):
            inbox.setdefault(agent_idx, {})[self.INPUT] = query
        return inbox

    def _normalize_outs(self, out: Any) -> List[str]:
        if isinstance(out, list):
            return [str(x) for x in out]
        if out is None:
            return []
        return [str(out)]

    def _build_initial_state(
        self,
        query: str,
        *,
        max_rounds: int,
        active_agents: Optional[Set[int]] = None,
        forced_outputs: Optional[Dict[int, List[str]]] = None,
    ) -> _MASState:
        return {
            "query": query,
            "inbox": self._seed_inbox_with_query(query),
            "agent_io": {},
            "primary_out": {},
            "executed": {},
            "active_agents": None if active_agents is None else {int(i) for i in active_agents},
            "forced_outputs": {
                int(agent_idx): self._normalize_outs(outs)
                for agent_idx, outs in (forced_outputs or {}).items()
            },
            "rounds": 0,
            "max_rounds": max(0, int(max_rounds)),
            "execution_order": [],
        }

    def _scheduler_node(self, state: _MASState):
        if state["rounds"] >= state["max_rounds"]:
            return Command(goto=END)

        active_agents = state["active_agents"]
        executed = state["executed"]
        inbox = state["inbox"]

        ready: List[int] = []
        for agent_idx in range(self.n):
            if agent_idx in executed:
                continue
            if active_agents is not None and agent_idx not in active_agents:
                continue
            have = inbox.get(agent_idx, {})
            if have and self.required_parents[agent_idx].issubset(set(have.keys())):
                ready.append(agent_idx)

        if not ready:
            return Command(goto=END)

        return Command(
            update={
                "rounds": state["rounds"] + 1,
                "execution_order": state["execution_order"] + ready,
            },
            goto=[self._agent_node_name(agent_idx) for agent_idx in ready],
        )

    def _agent_node(self, idx: int):
        def run(state: _MASState):
            have = state["inbox"].get(idx, {})
            ordered_parents = self.parent_order[idx]
            inputs = [have[parent_idx] for parent_idx in ordered_parents if parent_idx in have]

            if state["active_agents"] is not None:
                outs = self._normalize_outs(state["forced_outputs"].get(idx, []))
            else:
                outs = self.step(inputs, idx)

            delivered: Dict[int, Dict[int, str]] = {}
            outputs: Dict[int, str] = {}
            children = self.children.get(idx, [])
            if outs and len(outs) == len(children):
                pairs = zip(children, outs)
            elif outs:
                pairs = ((child_idx, outs[0]) for child_idx in children)
            else:
                pairs = ()

            for child_idx, msg in pairs:
                delivered.setdefault(child_idx, {})[idx] = msg
                outputs[child_idx] = msg

            update: Dict[str, Any] = {
                "executed": {idx: True},
                "agent_io": {
                    idx: {
                        "inputs": {p: have[p] for p in ordered_parents if p in have},
                        "outputs": outputs,
                    }
                },
            }
            if outs:
                update["primary_out"] = {idx: outs[0]}
            if delivered:
                update["inbox"] = delivered
            return update

        return run

    def _build_graph(self):
        graph = StateGraph(_MASState)
        graph.add_node("scheduler", self._scheduler_node)
        for agent_idx in range(self.n):
            graph.add_node(self._agent_node_name(agent_idx), self._agent_node(agent_idx))
            graph.add_edge(self._agent_node_name(agent_idx), "scheduler")
        graph.add_edge(START, "scheduler")
        return graph.compile()

    def _run_graph(
        self,
        query: str,
        *,
        max_rounds: int,
        active_agents: Optional[Set[int]] = None,
        forced_outputs: Optional[Dict[int, List[str]]] = None,
    ) -> _MASState:
        state = self._build_initial_state(
            query,
            max_rounds=max_rounds,
            active_agents=active_agents,
            forced_outputs=forced_outputs,
        )
        recursion_limit = max(4, 2 * max(0, int(max_rounds)) + self.n + 4)
        return self._graph.invoke(state, config={"recursion_limit": recursion_limit})

    def _last_primary_out(self, state: _MASState) -> str:
        last = state["query"]
        primary_out = state["primary_out"]
        for agent_idx in state["execution_order"]:
            if agent_idx in primary_out:
                last = primary_out[agent_idx]
        return last

    def _aggregate_final(self, primary_out: Dict[int, str], last: str) -> str:
        finals = [primary_out[i] for i in self.sinks if i in primary_out]
        if len(finals) == 1:
            return finals[0]
        if len(finals) > 1:
            ordered = [
                f"[agent {i}] {primary_out[i]}"
                for i in sorted(k for k in self.sinks if k in primary_out)
            ]
            return "\n\n".join(ordered)
        return last

    def _replay(
        self,
        query: str,
        active_agents: Set[int],
        forced_outputs: Dict[int, List[str]],
        *,
        include_agent_io: bool = False,
    ):
        state = self._run_graph(
            query,
            max_rounds=self.n,
            active_agents=active_agents,
            forced_outputs=forced_outputs,
        )
        last = self._last_primary_out(state)
        if include_agent_io:
            return state["inbox"], state["primary_out"], last, state["agent_io"]
        return state["inbox"], state["primary_out"], last

    # CHANGED: the second return value now records, for each agent, its inputs and outputs.
    def generate(
        self, query: str, max_steps: int = 64
    ) -> Tuple[str, Dict[int, Dict[str, Dict[int, str]]]]:
        state = self._run_graph(query, max_rounds=max_steps)
        last = self._last_primary_out(state)
        return self._aggregate_final(state["primary_out"], last), state["agent_io"]


def build_mas_from_specs(
    model,
    tok,
    agent_specs: List[Dict[str, Any]],
    edges: List[List[int]],
    *,
    use_openai: bool = False,
    openai_client=None,
    openai_model: str = "gpt-4.1-mini",
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    use_runpod: bool = False,
) -> MAS:
    """
    Build a MAS from config-provided agent specs and edges.
    Each agent spec supports:
      - system_prompt (str)
      - max_new_tokens (int, default 512)
    """
    agents = []
    for spec in agent_specs:
        agents.append(
            Agent(
                model,
                tok,
                system_prompt=spec.get("system_prompt", ""),
                max_new_tokens=int(spec.get("max_new_tokens", 512)),
                use_openai=use_openai,
                openai_client=openai_client,
                openai_model=openai_model,
                openai_api_key=openai_api_key,
                openai_base_url=openai_base_url,
                use_runpod=use_runpod,
            )
        )
    return MAS(edges, agents)
