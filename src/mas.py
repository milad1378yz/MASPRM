from typing import List, Tuple, Dict, Set
from agent import Agent


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

        self.edge_index = {e: i for i, e in enumerate(self.edges)}
        self.sinks = [i for i in range(self.n) if len(self.children.get(i, [])) == 0]

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

    # CHANGED: the second return value now records, for each agent, its inputs and outputs.
    def generate(
        self, query: str, max_steps: int = 64
    ) -> Tuple[str, Dict[int, Dict[str, Dict[int, str]]]]:
        # inbox[node][parent] = payload received from that parent
        inbox: Dict[int, Dict[int, str]] = {}

        # agent_io[agent] = {"inputs": {parent: msg}, "outputs": {child: msg}}
        agent_io: Dict[int, Dict[str, Dict[int, str]]] = {}

        # Parents required for each node (includes -1 if (-1, node) is in edges)
        required_parents: Dict[int, Set[int]] = {
            i: set(self.parents.get(i, [])) for i in range(self.n)
        }

        # Seed inputs from the external INPUT node to its children
        frontier: Set[int] = set(self.children.get(self.INPUT, []))
        for j in frontier:
            inbox.setdefault(j, {})[self.INPUT] = query

        # Only run nodes after they are fully ready; those ready now will be run in the first iteration.
        frontier = {j for j in frontier if required_parents[j].issubset(set(inbox[j].keys()))}

        executed: Set[int] = set()  # ensure each agent is called at most once
        primary_out: Dict[int, str] = (
            {}
        )  # representative single string per agent (for final aggregation)

        last = query
        steps = 0

        while frontier and steps < max_steps:
            next_frontier: Set[int] = set()

            for j in sorted(frontier):
                if j in executed:
                    continue

                have = inbox.get(j, {})
                # Order parents to form a deterministic input list
                ordered_parents = sorted(required_parents[j])
                inputs = [have[p] for p in ordered_parents if p in have]

                # Run the agent once with all its inputs
                outs = self.step(inputs, j)
                executed.add(j)

                # Record inputs/outputs by neighbor id
                agent_io[j] = {
                    "inputs": {p: have[p] for p in ordered_parents if p in have},
                    "outputs": {},
                }

                if outs:
                    primary_out[j] = outs[0]
                    last = outs[0]

                # Deliver outputs to children (pairwise if lengths match, else broadcast first)
                children = sorted(self.children.get(j, []))
                if children:
                    if outs and len(outs) == len(children):
                        pairs = zip(children, outs)
                    elif outs:
                        pairs = ((c, outs[0]) for c in children)
                    else:
                        pairs = ()

                    for c, msg in pairs:
                        inbox.setdefault(c, {})[j] = msg
                        agent_io[j]["outputs"][c] = msg

                    # Any child that just became fully ready is queued for the *next* iteration
                    for c in children:
                        if c not in executed:
                            req = required_parents[c]
                            if req.issubset(set(inbox.get(c, {}).keys())):
                                next_frontier.add(c)

            frontier = next_frontier
            steps += 1

        # Aggregate finals from sink nodes if available; otherwise fall back to the last produced text
        finals = [primary_out[i] for i in self.sinks if i in primary_out]
        if len(finals) == 1:
            return finals[0], agent_io
        if len(finals) > 1:
            ordered = [
                f"[agent {i}] {primary_out[i]}"
                for i in sorted(k for k in self.sinks if k in primary_out)
            ]
            return "\n\n".join(ordered), agent_io
        return last, agent_io
