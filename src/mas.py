import heapq
from typing import Annotated, Any, Dict, List, Optional, Set, Tuple, TypedDict

from agent import Agent
from langgraph.graph import START, StateGraph


def _merge_nested_dicts(
    left: Optional[Dict[int, Dict[int, str]]], right: Optional[Dict[int, Dict[int, str]]]
) -> Dict[int, Dict[int, str]]:
    merged: Dict[int, Dict[int, str]] = {int(k): dict(v) for k, v in (left or {}).items()}
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
    active_agents: Optional[Set[int]]
    forced_outputs: Dict[int, List[str]]
    max_rounds: int


class MAS:
    INPUT = -1  # special node id for the external query
    _INPUT_NODE = "mas_input"

    def __init__(self, edges: List[Tuple[int, int]], agents: List[Agent]):
        self.edges = [(int(s), int(t)) for s, t in edges]
        self.agents = agents
        self.n = len(agents)

        self.children: Dict[int, List[int]] = {}
        self.parents: Dict[int, List[int]] = {}
        for s, t in self.edges:
            if t < 0 or t >= self.n:
                raise ValueError(f"Invalid edge target {t}; expected an agent index in [0, {self.n}).")
            if s != self.INPUT and (s < 0 or s >= self.n):
                raise ValueError(
                    f"Invalid edge source {s}; expected {self.INPUT} or an agent index in [0, {self.n})."
                )
            self.children.setdefault(s, []).append(t)
            self.parents.setdefault(t, []).append(s)

        for node in self.children:
            self.children[node].sort()
        for node in self.parents:
            self.parents[node].sort()

        self.edge_index = {e: i for i, e in enumerate(self.edges)}
        self.agent_order = self._topological_order()
        self.agent_position = {agent_idx: pos for pos, agent_idx in enumerate(self.agent_order)}
        self.sinks = [i for i in range(self.n) if len(self.children.get(i, [])) == 0]
        self.required_parents: Dict[int, Set[int]] = {
            i: set(self.parents.get(i, [])) for i in range(self.n)
        }
        self.parent_order: Dict[int, List[int]] = {i: sorted(self.required_parents[i]) for i in range(self.n)}
        self.agent_round = self._compute_agent_rounds()
        self._graph = self._build_graph()

    def _topological_order(self) -> List[int]:
        in_degree = {i: 0 for i in range(self.n)}
        graph_children: Dict[int, List[int]] = {i: [] for i in range(self.n)}

        for s, t in self.edges:
            if s == self.INPUT:
                continue
            graph_children[s].append(t)
            in_degree[t] += 1

        ready = [agent_idx for agent_idx, degree in in_degree.items() if degree == 0]
        heapq.heapify(ready)

        order: List[int] = []
        while ready:
            agent_idx = heapq.heappop(ready)
            order.append(agent_idx)
            for child_idx in sorted(graph_children[agent_idx]):
                in_degree[child_idx] -= 1
                if in_degree[child_idx] == 0:
                    heapq.heappush(ready, child_idx)

        if len(order) != self.n:
            raise ValueError("MAS agent graph must be acyclic.")
        return order

    def _compute_agent_rounds(self) -> Dict[int, int]:
        rounds: Dict[int, int] = {}
        for agent_idx in self.agent_order:
            parent_rounds = [
                0 if parent_idx == self.INPUT else rounds[parent_idx]
                for parent_idx in self.parents.get(agent_idx, [])
            ]
            rounds[agent_idx] = (max(parent_rounds) + 1) if parent_rounds else 1
        return rounds

    def _agent_node_name(self, idx: int) -> str:
        return f"agent_{idx}"

    def _graph_node_name(self, idx: int) -> str:
        return self._INPUT_NODE if idx == self.INPUT else self._agent_node_name(idx)

    def _normalize_outs(self, out: Any) -> List[str]:
        if isinstance(out, list):
            return [str(x) for x in out]
        if out is None:
            return []
        return [str(out)]

    def agent_input_map(self, idx: int, inbox: Dict[int, Dict[int, str]]) -> Dict[int, str]:
        have = inbox.get(idx, {})
        return {parent_idx: have[parent_idx] for parent_idx in self.parent_order[idx] if parent_idx in have}

    def agent_inputs(self, idx: int, inbox: Dict[int, Dict[int, str]]) -> List[str]:
        return list(self.agent_input_map(idx, inbox).values())

    def agent_user_content(self, idx: int, inbox: Dict[int, Dict[int, str]]) -> str:
        return "\n\n".join(self.agent_inputs(idx, inbox)).strip()

    def agent_messages(self, idx: int, inbox: Dict[int, Dict[int, str]]):
        agent = self.agents[idx]
        user_content = self.agent_user_content(idx, inbox)
        if hasattr(agent, "format_messages"):
            return agent.format_messages(user_content)
        sys_prompt = getattr(agent, "system_prompt", "You are a helpful assistant.")
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]

    def step(self, messages: List[str], idx: int, **gen_kwargs) -> List[str]:
        user_content = "\n\n".join(messages).strip()
        agent = self.agents[idx]
        if hasattr(agent, "format_messages"):
            msgs = agent.format_messages(user_content)
        else:
            sys_prompt = getattr(agent, "system_prompt", "You are a helpful assistant.")
            msgs = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ]
        out = agent.generate(msgs, **gen_kwargs)
        return self._normalize_outs(out)

    def run_agent(self, idx: int, inbox: Dict[int, Dict[int, str]], **gen_kwargs) -> List[str]:
        out = self.agents[idx].generate(self.agent_messages(idx, inbox), **gen_kwargs)
        return self._normalize_outs(out)

    def sample_candidates(
        self,
        idx: int,
        inbox: Dict[int, Dict[int, str]],
        n_candidates: int = 2,
        **gen_kwargs,
    ) -> List[List[str]]:
        agent = self.agents[idx]
        msgs = self.agent_messages(idx, inbox)
        if hasattr(agent, "generate_n"):
            outs_list = agent.generate_n(msgs, n=n_candidates, **gen_kwargs)
        else:
            outs_list = [agent.generate(msgs, **gen_kwargs) for _ in range(n_candidates)]
        return [self._normalize_outs(out) for out in outs_list]

    def _deliver_outputs(self, idx: int, outs: List[str]) -> Tuple[Dict[int, Dict[int, str]], Dict[int, str]]:
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
        return delivered, outputs

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
            "inbox": {},
            "agent_io": {},
            "primary_out": {},
            "active_agents": None if active_agents is None else {int(i) for i in active_agents},
            "forced_outputs": {
                int(agent_idx): self._normalize_outs(outs)
                for agent_idx, outs in (forced_outputs or {}).items()
            },
            "max_rounds": max(0, int(max_rounds)),
        }

    def _input_node(self, state: _MASState):
        delivered = {
            agent_idx: {self.INPUT: state["query"]}
            for agent_idx in self.children.get(self.INPUT, [])
        }
        return {"inbox": delivered} if delivered else {}

    def _agent_node(self, idx: int):
        def run(state: _MASState):
            active_agents = state["active_agents"]
            if active_agents is not None and idx not in active_agents:
                return {}
            if self.agent_round[idx] > state["max_rounds"]:
                return {}

            have = state["inbox"].get(idx, {})
            if not have or not self.required_parents[idx].issubset(set(have.keys())):
                return {}

            if active_agents is not None:
                outs = self._normalize_outs(state["forced_outputs"].get(idx, []))
            else:
                outs = self.run_agent(idx, state["inbox"])

            delivered, outputs = self._deliver_outputs(idx, outs)
            update: Dict[str, Any] = {
                "agent_io": {
                    idx: {
                        "inputs": self.agent_input_map(idx, state["inbox"]),
                        "outputs": outputs,
                    }
                }
            }
            if outs:
                update["primary_out"] = {idx: outs[0]}
            if delivered:
                update["inbox"] = delivered
            return update

        return run

    def _build_graph(self):
        graph = StateGraph(_MASState)
        graph.add_node(self._INPUT_NODE, self._input_node)
        for agent_idx in range(self.n):
            graph.add_node(self._agent_node_name(agent_idx), self._agent_node(agent_idx))

        graph.add_edge(START, self._INPUT_NODE)
        for agent_idx in self.agent_order:
            parents = [self._graph_node_name(parent_idx) for parent_idx in self.parent_order[agent_idx]]
            if not parents:
                continue
            if len(parents) == 1:
                graph.add_edge(parents[0], self._agent_node_name(agent_idx))
            else:
                graph.add_edge(parents, self._agent_node_name(agent_idx))
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
        recursion_limit = max(self.n + 4, int(max_rounds) + 4)
        return self._graph.invoke(state, config={"recursion_limit": recursion_limit})

    def _last_primary_out(self, state: _MASState) -> str:
        last = state["query"]
        primary_out = state["primary_out"]
        for agent_idx in self.agent_order:
            if agent_idx in primary_out:
                last = primary_out[agent_idx]
        return last

    def _aggregate_final(self, primary_out: Dict[int, str], last: str) -> str:
        finals = [primary_out[i] for i in self.sinks if i in primary_out]
        if len(finals) == 1:
            return finals[0]
        if len(finals) > 1:
            ordered = [f"[agent {i}] {primary_out[i]}" for i in self.sinks if i in primary_out]
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
            )
        )
    return MAS(edges, agents)
