import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from answer_utils import is_correct
from mas import MAS
from tqdm import tqdm


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
    init_value: float = field(default=0.0, repr=False)
    init_visits: int = field(default=0, repr=False)
    prior: float = field(default=0.0, repr=False)

    @property
    def q_mean(self) -> float:
        denom = self.visits + self.init_visits
        if denom <= 0:
            return 0.0
        return (self.q_sum + self.init_value) / denom


class BaseMCTS:
    def __init__(
        self,
        mas: MAS,
        question: str,
        *,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        root_init_value: float = 0.0,
        root_init_visits: int = 0,
    ):
        self.mas = mas
        self.question = question
        self.gen_kwargs = gen_kwargs or {}
        self.order = list(self.mas.agent_order)
        self.required_parents: Dict[int, Set[int]] = {
            i: set(self.mas.parents.get(i, [])) for i in range(self.mas.n)
        }
        self.root = Node(
            steps=[],
            trajectory={},
            node_text=self.question,
            init_value=root_init_value,
            init_visits=root_init_visits,
        )

    def _depth_of(self, node: Node) -> int:
        return len(node.trajectory)

    def _is_terminal(self, node: Node) -> bool:
        return self._depth_of(node) >= len(self.order)

    def _expansion_context(self, node: Node) -> tuple[int, Dict[int, Dict[int, str]]]:
        depth = self._depth_of(node)
        if depth >= len(self.order):
            raise RuntimeError("Tried to expand a terminal MCTS node.")

        agent_idx = self.order[depth]
        inbox, _, _ = self.mas._replay(self.question, set(node.trajectory), node.trajectory)

        req = self.required_parents[agent_idx]
        have = set(inbox.get(agent_idx, {}).keys())
        if not req.issubset(have):
            missing = sorted(req - have)
            raise RuntimeError(
                f"Agent {agent_idx} cannot run; missing parents {missing}. "
                "Check edges and per-depth ordering."
            )
        return agent_idx, inbox

    def _sample_candidates(self, node: Node, n_candidates: int) -> tuple[int, List[List[str]]]:
        agent_idx, inbox = self._expansion_context(node)
        candidates = self.mas.sample_candidates(
            agent_idx,
            inbox,
            n_candidates=n_candidates,
            **self.gen_kwargs,
        )
        return agent_idx, candidates

    def _finalize_node(self, node: Node) -> None:
        if not self._is_terminal(node):
            return
        if node.final_answer is None:
            inbox, primary_out, last = self.mas._replay(
                self.question,
                set(node.trajectory),
                node.trajectory,
            )
            node.final_answer = self.mas._aggregate_final(primary_out, last)
        node.is_terminal = True

    def _make_child(
        self,
        parent: Node,
        agent_idx: int,
        outs: List[str],
        *,
        init_value: float = 0.0,
        init_visits: int = 0,
        prior: float = 0.0,
    ) -> Node:
        child = Node(
            steps=parent.steps + ([outs[0]] if outs else [""]),
            action_text=f"[agent {agent_idx}] {outs[0] if outs else ''}",
            is_terminal=False,
            final_answer=None,
            node_text=(outs[0] if outs else ""),
            trajectory={**parent.trajectory, agent_idx: outs},
            init_value=init_value,
            init_visits=init_visits,
            prior=prior,
        )
        self._finalize_node(child)
        return child


def uct_score(parent: Node, child: Node, c: float) -> float:
    Np = max(parent.visits, 1)
    Nc = max(child.visits, 1)
    return child.q_mean + c * math.sqrt(math.log(Np) / Nc)


class MAS_MCTS(BaseMCTS):
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
        super().__init__(mas, question, gen_kwargs=gen_kwargs)
        self.truth = ground_truth_answer
        self.c = c
        self.n_candidates = n_candidates

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
                self._finalize_node(node)
                return path

            # Expand if no children
            if not node.children:
                depth = self._depth_of(node)  # which agent to decide
                agent_idx, candidates = self._sample_candidates(
                    node,
                    self.n_candidates
                    if depth < (len(self.order) - 1)
                    else max(1, self.n_candidates // 2),
                )

                for outs in candidates:
                    node.children.append(self._make_child(node, agent_idx, outs))

                if not node.children:
                    return path

                # Immediately grade all terminal children so none remain unvisited
                terminals = [ch for ch in node.children if ch.is_terminal]
                if terminals:
                    # Choose one terminal child to continue this rollout's path
                    chosen = random.choice(terminals)
                    # Backpropagate on the other terminal siblings right now
                    for sib in terminals:
                        if sib is not chosen:
                            # immediate backprop
                            self._backprop(path + [sib])
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
