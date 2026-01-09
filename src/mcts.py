import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from answer_utils import (
    normalize_answer,
    extract_final_letter,
    answers_equal,
    _to_number,
    extract_pred_number,
    eq,
    numbers_equal,
    is_correct,
)
from mas import MAS
from tqdm import tqdm


def _normalize_to_list(out: Any) -> List[str]:
    if isinstance(out, list):
        return [str(x) for x in out]
    return [str(out)]


def propose_agent_candidates(
    mas: MAS,
    agent_idx: int,
    inbox: Dict[int, Dict[int, str]],
    required_parents: Dict[int, Set[int]],
    n_candidates: int = 2,
    **gen_kwargs,
) -> List[List[str]]:
    """
    Sample multiple candidate outputs for agent `agent_idx` given current inbox.
    Returns a list of candidates, where each candidate = List[str] (one per child or broadcast).
    """
    have = inbox.get(agent_idx, {})
    ordered_parents = sorted(required_parents[agent_idx])
    inputs = [have[p] for p in ordered_parents if p in have]

    # Build chat messages (mirror MAS.step)
    sys_prompt = getattr(mas.agents[agent_idx], "system_prompt", "You are a helpful assistant.")
    content = "\n\n".join(inputs).strip()
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": content},
    ]

    agent = mas.agents[agent_idx]

    # If Agent supports generate_n, use it; else call generate repeatedly.
    outs_list: List[Any]
    if hasattr(agent, "generate_n"):
        outs_list = agent.generate_n(msgs, n=n_candidates, **gen_kwargs)
    else:
        outs_list = [agent.generate(msgs, **gen_kwargs) for _ in range(n_candidates)]

    # Normalize each into List[str]
    normd: List[List[str]] = []
    for out in outs_list:
        normd.append(_normalize_to_list(out))
    return normd


def _seed_inbox_with_query(mas: MAS, query: str) -> Dict[int, Dict[int, str]]:
    """inbox[node][parent] = payload received from that parent."""
    inbox: Dict[int, Dict[int, str]] = {}
    for j in mas.children.get(MAS.INPUT, []):
        inbox.setdefault(j, {})[MAS.INPUT] = query
    return inbox


def _deliver_outputs(mas: MAS, agent_idx: int, outs: List[str], inbox: Dict[int, Dict[int, str]]):
    """Deliver outs to children using the same broadcast/pairwise rule as MAS.generate."""
    children = sorted(mas.children.get(agent_idx, []))
    if not children:
        return
    if outs and len(outs) == len(children):
        pairs = zip(children, outs)
    elif outs:
        pairs = ((c, outs[0]) for c in children)
    else:
        pairs = ()
    for c, msg in pairs:
        inbox.setdefault(c, {})[agent_idx] = msg


def _replay_trajectory(mas: MAS, query: str, trajectory: Dict[int, List[str]]):
    """
    Recompute inbox and primary_out given a trajectory {agent: outs}.
    Also return `last` (last primary text produced, for fallback).
    """
    inbox = _seed_inbox_with_query(mas, query)
    primary_out: Dict[int, str] = {}
    last = query
    for j in range(mas.n):
        if j not in trajectory:
            continue
        outs = trajectory[j]
        if outs:
            primary_out[j] = outs[0]
            last = outs[0]
        _deliver_outputs(mas, j, outs, inbox)
    return inbox, primary_out, last


def _aggregate_final(mas: MAS, primary_out: Dict[int, str], last: str) -> str:
    finals = [primary_out[i] for i in mas.sinks if i in primary_out]
    if len(finals) == 1:
        return finals[0]
    if len(finals) > 1:
        ordered = [
            f"[agent {i}] {primary_out[i]}"
            for i in sorted(k for k in mas.sinks if k in primary_out)
        ]
        return "\n\n".join(ordered)
    return last


# Answer parsing/comparison utilities live in answer_utils.py (re-exported here for compatibility).

@dataclass
class Node:
    steps: List[str]  # here: per-agent primary texts chosen so far
    action_text: Optional[str] = None
    is_terminal: bool = False
    final_answer: Optional[str] = None
    q_sum: float = 0.0
    visits: int = 0
    children: List["Node"] = field(default_factory=list)
    # Internal state (NOT serialized):
    trajectory: Dict[int, List[str]] = field(
        default_factory=dict, repr=False
    )  # agent -> outs (list[str])
    node_text: Optional[str] = None

    @property
    def q_mean(self) -> float:
        return self.q_sum / self.visits if self.visits > 0 else 0.0


def uct_score(parent: Node, child: Node, c: float) -> float:
    Np = max(parent.visits, 1)
    Nc = max(child.visits, 1)
    return child.q_mean + c * math.sqrt(math.log(Np) / Nc)


class MAS_MCTS:
    def __init__(
        self,
        mas: MAS,
        question: str,
        ground_truth_answer: str,
        c: float = 2.0,
        n_candidates: int = 4,
        gen_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        mas: your MAS instance
        question: the external query
        ground_truth_answer: string to "hard-check" leaves
        c: UCT exploration constant
        n_candidates: number of per-agent samples to generate at expansion
        gen_kwargs: forwarded to Agent.generate(_n), e.g., temperature, top_p, max_new_tokens
        """
        self.mas = mas
        self.question = question
        self.truth = ground_truth_answer
        self.c = c
        self.n_candidates = n_candidates
        self.gen_kwargs = gen_kwargs or {}

        # Precompute required parents per agent (includes -1 if present)
        self.required_parents: Dict[int, Set[int]] = {
            i: set(self.mas.parents.get(i, [])) for i in range(self.mas.n)
        }

        # (Optional) sanity check: all agent parents must be subset of {INPUT} ∪ {previous agents}
        for j in range(self.mas.n):
            invalid = [p for p in self.required_parents[j] if p not in {-1} and p >= j]
            if invalid:
                raise ValueError(
                    f"Agent {j} has forward/invalid parents {invalid}. "
                    "Ensure edges are topologically ordered (no future parents)."
                )

        self.root = Node(steps=[], trajectory={}, node_text=self.question)

    def _depth_of(self, node: Node) -> int:
        """How many agents have been decided on the path to this node."""
        return len(node.trajectory)

    def _is_terminal(self, node: Node) -> bool:
        return self._depth_of(node) >= self.mas.n

    def search(
        self,
        n_rollouts: int = 32,
        early_stop_patience: int = 24,
        tol: float = 1e-2,  # treat changes smaller than this as "no change"
    ) -> Node:
        """
        Run MCTS rollouts, stopping early if the root q estimate stops changing
        (increase OR decrease) for `early_stop_patience` consecutive iterations.
        Assumes q values are in [-1, 1].
        """
        last = None
        stale = 0

        for _ in tqdm(range(n_rollouts), desc="MCTS rollouts"):
            path = self._rollout(self.root)
            self._backprop(path)

            if not self.root.children:
                continue

            cur = self.root.q_mean

            if last is None or abs(cur - last) > tol:
                last = cur
                if cur >= 1.0 - 0.05 or cur <= -1.0 + 0.05:
                    stale += 1
                else:
                    stale = 0
            else:
                stale += 1

            if stale >= early_stop_patience:
                break

        return self.root

    def _rollout(self, root: Node) -> List[Node]:
        node = root
        path = [node]

        while True:
            if self._is_terminal(node):
                # Compute final answer if not already present
                if node.final_answer is None:
                    inbox, primary_out, last = _replay_trajectory(
                        self.mas, self.question, node.trajectory
                    )
                    node.final_answer = _aggregate_final(self.mas, primary_out, last)
                    node.is_terminal = True
                return path

            # Expand if no children
            if not node.children:
                depth = self._depth_of(node)  # which agent to decide
                agent_idx = depth  # 0-based
                # Recompute inbox given all earlier agent choices
                inbox, _, _ = _replay_trajectory(self.mas, self.question, node.trajectory)

                # Ensure required inputs are present for agent_idx
                req = self.required_parents[agent_idx]
                have = set(inbox.get(agent_idx, {}).keys())
                if not req.issubset(have):
                    # If this happens, topology is inconsistent with depth ordering
                    missing = sorted(req - have)
                    raise RuntimeError(
                        f"Agent {agent_idx} cannot run; missing parents {missing}. "
                        "Check edges and per-depth ordering."
                    )

                # Propose multiple candidates for this agent
                n_cand = (
                    self.n_candidates
                    if agent_idx < (self.mas.n - 1)
                    else max(1, self.n_candidates // 2)
                )
                candidates = propose_agent_candidates(
                    self.mas,
                    agent_idx,
                    inbox,
                    self.required_parents,
                    n_candidates=n_cand,
                    **self.gen_kwargs,
                )

                # Create children, one per candidate
                for outs in candidates:
                    child = Node(
                        steps=node.steps + ([outs[0]] if outs else [""]),
                        action_text=f"[agent {agent_idx}] {outs[0] if outs else ''}",
                        is_terminal=False,
                        final_answer=None,
                        node_text=(outs[0] if outs else ""),
                        trajectory={**node.trajectory, agent_idx: outs},
                    )

                    # If this child already completed all agents, make it terminal now
                    if self._depth_of(child) >= self.mas.n:
                        inbox2, primary_out2, last2 = _replay_trajectory(
                            self.mas, self.question, child.trajectory
                        )
                        child.final_answer = _aggregate_final(self.mas, primary_out2, last2)
                        child.is_terminal = True

                    node.children.append(child)

                # Immediately grade all terminal children so none remain unvisited
                terminals = [ch for ch in node.children if ch.is_terminal]
                if terminals:
                    # Choose one terminal child to continue this rollout's path
                    ###################################################################################
                    chosen = random.choice(terminals)
                    # Backpropagate on the other terminal siblings right now
                    for sib in terminals:
                        if sib is not chosen:
                            # immediate backprop
                            self._backprop(path + [sib])

                    
                    ###################################################################################


                                        # # True:
                    # for ch in terminals:
                    #     # Ensure final_answer is computed
                    #     if ch.final_answer is None:
                    #         inbox2, primary_out2, last2 = _replay_trajectory(
                    #             self.mas, self.question, ch.trajectory
                    #         )
                    #         ch.final_answer = _aggregate_final(self.mas, primary_out2, last2)
                    #         ch.is_terminal = True

                    #     # "Soon backprop" **locally**: initialize the leaf’s own stats
                    #     reward = 1.0 if is_correct(ch.final_answer, self.truth) else -1.0
                    #     ch.visits = 1
                    #     ch.q_sum = reward

                    # # For this rollout, pick exactly ONE terminal child and continue path
                    # chosen = random.choice(terminals)
                    ###################################################################################
                    node = chosen
                    path.append(node)
                    continue

                # Otherwise (no terminal children), pick one unvisited child at random to continue
                node = random.choice(node.children)
                path.append(node)
                continue

            # Selection with UCT among existing children
            uct_vals = [uct_score(node, ch, self.c) for ch in node.children]
            best_idx = max(range(len(uct_vals)), key=lambda i: uct_vals[i])
            node = node.children[best_idx]
            path.append(node)

    def _backprop(self, path: List[Node]):
        leaf = path[-1]
        # terminal-guided reward
        if leaf.is_terminal and leaf.final_answer is not None:
            reward = 1.0 if is_correct(leaf.final_answer, self.truth) else -1.0
        else:
            reward = -1.0

        for n in path:
            n.visits += 1
            n.q_sum += reward
