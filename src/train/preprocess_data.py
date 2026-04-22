"""
Example:
    # PRM-only (original behavior, global view)
    python src/train/preprocess_data.py --target-nodes 4 --dedupe

    # PPM (global view)
    python preprocess_prm.py \
        --input trees_dir_or_file \
        --make-ppm \
        --ppm-max-pairs 8

    # Local view (MASPRM: each record scoped to a single agent's inbox + output)
    python src/train/preprocess_data.py --view local --make-ppm
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from tqdm import tqdm

import yaml


_ACTION_TEXT_RE = re.compile(r"^\[agent\s+(\d+)\]")


def _parse_agent_idx(action_text: Optional[str]) -> Optional[int]:
    """Recover the agent index from a tree node's action_text (set by MCTS).

    Nodes store `action_text = f"[agent {idx}] {output}"`; for the root and
    anything malformed, return None.
    """
    if not isinstance(action_text, str):
        return None
    m = _ACTION_TEXT_RE.match(action_text)
    return int(m.group(1)) if m else None


def _derive_config_path(input_dir: Path) -> Path:
    """Map `results/<dataset>_res_<...>` → `configs/<dataset>.yaml`."""
    dataset = input_dir.name.split("_res_")[0]
    return Path("configs") / f"{dataset}.yaml"


def _build_stub_mas(config_path: Path):
    """Build a model-less MAS used only for inbox routing.

    MAS._replay with `active_agents` set uses forced_outputs and never calls
    agent.generate, so agents don't need a real model or tokenizer.
    """
    # Ensure src/ is importable for `mas` and `agent` modules.
    src_root = Path(__file__).resolve().parent.parent
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from mas import MAS  # noqa: E402
    from agent import Agent  # noqa: E402

    cfg = yaml.safe_load(Path(config_path).read_text())
    agent_specs = cfg["agents"]
    edges = cfg["edges"]
    agents = [
        Agent(
            model=None,
            tok=None,
            system_prompt=spec.get("system_prompt", "You are a helpful assistant."),
            max_new_tokens=int(spec.get("max_new_tokens", 512)),
        )
        for spec in agent_specs
    ]
    return MAS(edges, agents)


def _node_value(nd: Dict[str, Any]) -> Optional[float]:
    """Return q_mean if present; otherwise compute q_sum/visits. None if unknown."""
    q = nd.get("q_mean", None)
    if q is not None:
        try:
            return float(q)
        except Exception:
            return None
    # fallback
    try:
        v = float(nd.get("q_sum", 0.0))
        n = int(nd.get("visits", 0))
        return (v / n) if n > 0 else None
    except Exception:
        return None


def _is_pm1(x: Optional[float]) -> bool:
    return (x is not None) and math.isclose(abs(float(x)), 1.0, rel_tol=0.0, abs_tol=1e-6)


def _iter_paths_exact_nodes(
    root: Dict[str, Any], target_nodes: int
) -> Iterable[List[Dict[str, Any]]]:
    """
    DFS that yields paths having EXACTLY `target_nodes` nodes (root included).
    """
    stack: List[tuple[Dict[str, Any], List[Dict[str, Any]]]] = [(root, [root])]
    while stack:
        nd, path = stack.pop()
        if len(path) == target_nodes:
            yield path
            continue
        children = nd.get("children") or []
        for ch in reversed(children):
            stack.append((ch, path + [ch]))


def _extract_prompt(root: Dict[str, Any], fallback: str) -> str:
    txt = root.get("node_text")
    if isinstance(txt, str) and txt.strip():
        return txt.strip()
    return fallback  # safe fallback to filename


def _actions_and_labels_from_path(
    path: List[Dict[str, Any]],
) -> Optional[Tuple[List[str], List[float]]]:
    """
    Build (actions, labels) for PRM from a root->leaf path (root excluded).
    Returns None if any step is invalid.
    """
    actions: List[str] = []
    labels: List[float] = []
    for nd in path[1:]:
        step_txt = (nd.get("node_text") or "").strip()
        if not step_txt:
            return None
        q = _node_value(nd)
        if q is None:  # must have a target at each step
            return None
        actions.append(step_txt)
        # PRM expects targets in [-1, 1]
        labels.append(float(max(-1.0, min(1.0, q))))
    return actions, labels


def _records_from_tree(
    data: Dict[str, Any],
    *,
    filename_hint: str,
    target_nodes: int = 5,  # root + 4 actions
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Original PRM-sample extraction (kept intact for backward compatibility).
    """
    prompt = _extract_prompt(data, fallback=filename_hint)

    # FILTER: keep only trees whose ROOT score is within [-0.7, 0.7]
    root_q = _node_value(data)
    if (root_q is None) or not (-0.7 <= float(root_q) <= 0.7):
        return [], 0, 0

    out: List[Dict[str, Any]] = []
    n_pos = 0  # number of terminal leaves with reward +1
    n_neg = 0  # number of terminal leaves with reward -1

    for path in _iter_paths_exact_nodes(data, target_nodes=target_nodes):
        leaf = path[-1]

        # Only accept terminal leaves with q == ±1
        if not bool(leaf.get("is_terminal", False)):
            continue
        leaf_q = _node_value(leaf)
        if not _is_pm1(leaf_q):
            continue

        res = _actions_and_labels_from_path(path)
        if res is None:
            continue
        actions, labels = res

        # Exactly target_nodes-1 actions/labels expected
        if len(actions) != (target_nodes - 1) or len(labels) != (target_nodes - 1):
            continue

        # Count sign of the terminal reward (leaf)
        if float(leaf_q) > 0:
            n_pos += 1
        else:
            n_neg += 1

        out.append({"prompt": prompt, "completions": actions, "labels": labels})

    return out, n_pos, n_neg


def _records_from_tree_local(
    data: Dict[str, Any],
    *,
    mas: Any,
    filename_hint: str,
    target_nodes: int = 5,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Local-view PRM records: one record per (path, step).

    Each record's `prompt` is the current agent's replayed inbox (i.e., the
    concatenation of messages routed to it by MAS given the prior decisions),
    and `completions=[node_text]` with `labels=[q]` — exactly the scope seen
    by `render_state_text(view="local")` at inference time.
    """
    question = _extract_prompt(data, fallback=filename_hint)

    # Preserve the same root-q filter as the global path.
    root_q = _node_value(data)
    if (root_q is None) or not (-0.7 <= float(root_q) <= 0.7):
        return [], 0, 0

    out: List[Dict[str, Any]] = []
    n_pos = 0
    n_neg = 0

    for path in _iter_paths_exact_nodes(data, target_nodes=target_nodes):
        leaf = path[-1]
        if not bool(leaf.get("is_terminal", False)):
            continue
        leaf_q = _node_value(leaf)
        if not _is_pm1(leaf_q):
            continue

        # Recover per-step (agent_idx, output, q) for the path (root excluded).
        step_info: List[Tuple[int, str, float]] = []
        ok = True
        for nd in path[1:]:
            aidx = _parse_agent_idx(nd.get("action_text"))
            if aidx is None:
                ok = False
                break
            txt = (nd.get("node_text") or "").strip()
            if not txt:
                ok = False
                break
            q = _node_value(nd)
            if q is None:
                ok = False
                break
            step_info.append((aidx, txt, max(-1.0, min(1.0, float(q)))))
        if not ok or len(step_info) != (target_nodes - 1):
            continue

        if float(leaf_q) > 0:
            n_pos += 1
        else:
            n_neg += 1

        # Emit one local record per step, replaying only the prior decisions
        # to recover the agent's inbox at that point in the routed DAG.
        for i, (aidx, output, q) in enumerate(step_info):
            prior_forced = {a: [t] for (a, t, _) in step_info[:i]}
            try:
                inbox, _, _ = mas._replay(question, set(prior_forced), prior_forced)
            except Exception:
                continue
            user_content = mas.agent_user_content(aidx, inbox)
            if not user_content:
                # Entry agent with no routed parents (shouldn't happen once the
                # INPUT edge is set, but guard anyway).
                user_content = question
            out.append(
                {
                    "prompt": user_content,
                    "completions": [output],
                    "labels": [q],
                    "meta": {
                        "agent_idx": aidx,
                        "step_index": i,
                        "target_nodes": target_nodes,
                        "view": "local",
                    },
                }
            )

    return out, n_pos, n_neg


def _ppm_pairs_from_tree(
    data: Dict[str, Any],
    *,
    filename_hint: str,
    target_nodes: int,
    pos_topk: int = 4,
    neg_topk: int = 4,
    max_pairs: int = 8,
    require_root_window: Optional[Tuple[float, float]] = None,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    rStar-Math style step-wise PPM (corrected):

      • For EACH prefix node at depth d (0 <= d < target_nodes-1),
        form pairs of next actions chosen from disjoint top-K and bottom-K
        *unique-by-text* children ranked by Q.
      • Pairs share the same preceding steps (path[1:d+1]) and differ at
        EXACTLY one step (the current child).
      • For the FINAL step (d == target_nodes-2), only terminal children
        are considered.
      • max_pairs limits the TOTAL pairs per tree.

    Returns: (records, num_positive_candidates, num_negative_candidates).
    """
    # Optional root-q filter
    if require_root_window is not None:
        root_q = _node_value(data)
        lo, hi = require_root_window
        if (root_q is None) or not (float(lo) <= float(root_q) <= float(hi)):
            return []

    prompt = _extract_prompt(data, fallback=filename_hint)

    def _norm_text(s: str) -> str:
        # Collapse whitespace and strip; keeps only text-dedup, not formatting in output
        return " ".join(s.split())

    out: List[Dict[str, Any]] = []
    seen_pairs: set[Tuple[Tuple[str, ...], Tuple[str, ...]]] = set()

    pairs_made_total = 0

    MIN_MARGIN = 1e-9  # require pos_q > neg_q + MIN_MARGIN

    # DFS stack over prefix nodes
    stack: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = [(data, [data])]
    while stack and (pairs_made_total < max_pairs):
        nd, path = stack.pop()
        depth = len(path) - 1  # root=0

        # Enqueue deeper nodes (so we can consider their prefixes too)
        if depth < target_nodes - 1:
            for ch in reversed(nd.get("children") or []):
                stack.append((ch, path + [ch]))

        # Only form pairs if there IS a next step to choose
        if depth >= target_nodes - 1:
            continue

        # Children usable at this step
        children = nd.get("children") or []
        # For final step, restrict to terminal children only
        if depth == target_nodes - 2:
            children = [c for c in children if bool(c.get("is_terminal", False))]
        if not children:
            continue

        # Collect (normalized_text, original_text, child_node, q)
        scored: List[Tuple[str, str, Dict[str, Any], float]] = []
        for ch in children:
            raw = (ch.get("node_text") or "").strip()
            if not raw:
                continue
            q = _node_value(ch)
            if q is None:
                continue
            scored.append((_norm_text(raw), raw, ch, float(q)))

        if not scored:
            continue

        # Keep only one candidate per unique normalized text (the one with highest Q)
        best_by_text: Dict[str, Tuple[str, Dict[str, Any], float]] = {}
        # map: norm_text -> (original_text, node, q)
        for norm, raw, ch, q in scored:
            cur = best_by_text.get(norm)
            if (cur is None) or (q > cur[2]):
                best_by_text[norm] = (raw, ch, q)

        uniq: List[Tuple[str, Dict[str, Any], float]] = [
            (raw, ch, q) for (raw, ch, q) in best_by_text.values()
        ]
        if len(uniq) < 2:
            # Need at least two distinct action texts to form a pair
            continue

        # Sort by Q ascending; then split into bottom-K and top-K WITHOUT overlap
        uniq.sort(key=lambda t: t[2])  # (raw_text, node, q)
        # Choose K- on the left, K+ on the right with no intersection
        k_neg = min(neg_topk, max(1, len(uniq) // 2))
        k_pos = min(pos_topk, max(1, len(uniq) - k_neg))
        # ensure non-overlap
        if k_neg + k_pos > len(uniq):
            k_pos = max(1, len(uniq) - k_neg)
        negs = uniq[:k_neg]
        poss = uniq[-k_pos:]

        if not negs or not poss:
            continue

        # Build the shared prefix (root excluded)
        prefix_actions: List[str] = []
        ok_prefix = True
        for node in path[1:]:
            s = (node.get("node_text") or "").strip()
            if not s:
                ok_prefix = False
                break
            prefix_actions.append(s)
        if not ok_prefix and depth > 0:
            continue

        # Cross product, filtered by margin and capped globally
        for p_raw, _, pq in reversed(poss):  # iterate high to low on positive side
            if pairs_made_total >= max_pairs:
                break
            for n_raw, _, nq in negs:
                if pairs_made_total >= max_pairs:
                    break
                # ensure strictly better positive
                if pq <= nq + MIN_MARGIN:
                    continue
                # texts must differ (should already hold due to uniq-by-text)
                if p_raw == n_raw:
                    continue

                chosen_actions = prefix_actions + [p_raw]
                rejected_actions = prefix_actions + [n_raw]

                key = (tuple(chosen_actions), tuple(rejected_actions))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)

                rec = {
                    "prompt": prompt,
                    "chosen_actions": chosen_actions,
                    "rejected_actions": rejected_actions,
                    "meta": {
                        "step_index": depth,  # 0-based (0 means first action)
                        "shared_prefix_len": depth,  # number of identical preceding steps
                        "target_nodes": target_nodes,
                        "parent_visits": int(nd.get("visits", 0)),
                        "pos_q": float(pq),
                        "neg_q": float(nq),
                        "q_gap": float(pq - nq),
                        "final_step_only_terminal": (depth == target_nodes - 2),
                    },
                }
                out.append(rec)
                pairs_made_total += 1

    return out


def _ppm_pairs_from_tree_local(
    data: Dict[str, Any],
    *,
    mas: Any,
    filename_hint: str,
    target_nodes: int,
    pos_topk: int = 4,
    neg_topk: int = 4,
    max_pairs: int = 8,
    require_root_window: Optional[Tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """Local-view PPM pairs: each pair differs at one agent's step output, with
    the shared upstream prefix collapsed into `prompt` via MAS inbox replay.
    """
    if require_root_window is not None:
        root_q = _node_value(data)
        lo, hi = require_root_window
        if (root_q is None) or not (float(lo) <= float(root_q) <= float(hi)):
            return []

    question = _extract_prompt(data, fallback=filename_hint)

    def _norm_text(s: str) -> str:
        return " ".join(s.split())

    out: List[Dict[str, Any]] = []
    seen_pairs: set = set()
    pairs_made_total = 0
    MIN_MARGIN = 1e-9

    # DFS stack carries (node, node_path, agent_idx_path). agent_idx_path
    # mirrors node_path[1:] (root excluded) and lets us build prefix_forced
    # without re-parsing action_text each time.
    stack: List[Tuple[Dict[str, Any], List[Dict[str, Any]], List[int]]] = [(data, [data], [])]
    while stack and (pairs_made_total < max_pairs):
        nd, path, agent_prefix = stack.pop()
        depth = len(path) - 1

        if depth < target_nodes - 1:
            for ch in reversed(nd.get("children") or []):
                child_aidx = _parse_agent_idx(ch.get("action_text"))
                if child_aidx is None:
                    continue
                stack.append((ch, path + [ch], agent_prefix + [child_aidx]))

        if depth >= target_nodes - 1:
            continue

        children = nd.get("children") or []
        if depth == target_nodes - 2:
            children = [c for c in children if bool(c.get("is_terminal", False))]
        if not children:
            continue

        # MCTS expands one agent per node, so all children share an agent_idx.
        current_agent_idx = _parse_agent_idx(children[0].get("action_text"))
        if current_agent_idx is None:
            continue

        scored: List[Tuple[str, str, Dict[str, Any], float]] = []
        for ch in children:
            raw = (ch.get("node_text") or "").strip()
            if not raw:
                continue
            q = _node_value(ch)
            if q is None:
                continue
            scored.append((_norm_text(raw), raw, ch, float(q)))
        if not scored:
            continue

        best_by_text: Dict[str, Tuple[str, Dict[str, Any], float]] = {}
        for norm, raw, ch, q in scored:
            cur = best_by_text.get(norm)
            if (cur is None) or (q > cur[2]):
                best_by_text[norm] = (raw, ch, q)

        uniq = [(raw, ch, q) for (raw, ch, q) in best_by_text.values()]
        if len(uniq) < 2:
            continue

        uniq.sort(key=lambda t: t[2])
        k_neg = min(neg_topk, max(1, len(uniq) // 2))
        k_pos = min(pos_topk, max(1, len(uniq) - k_neg))
        if k_neg + k_pos > len(uniq):
            k_pos = max(1, len(uniq) - k_neg)
        negs = uniq[:k_neg]
        poss = uniq[-k_pos:]
        if not negs or not poss:
            continue

        # Build prefix trajectory (agent_idx -> [text]) and replay inboxes.
        prefix_forced: Dict[int, List[str]] = {}
        ok_prefix = True
        for nd_p, aidx_p in zip(path[1:], agent_prefix):
            s = (nd_p.get("node_text") or "").strip()
            if not s:
                ok_prefix = False
                break
            prefix_forced[aidx_p] = [s]
        if not ok_prefix:
            continue

        try:
            inbox, _, _ = mas._replay(question, set(prefix_forced), prefix_forced)
        except Exception:
            continue
        user_content = mas.agent_user_content(current_agent_idx, inbox)
        if not user_content:
            user_content = question

        for p_raw, _, pq in reversed(poss):
            if pairs_made_total >= max_pairs:
                break
            for n_raw, _, nq in negs:
                if pairs_made_total >= max_pairs:
                    break
                if pq <= nq + MIN_MARGIN:
                    continue
                if p_raw == n_raw:
                    continue

                key = (user_content, p_raw, n_raw)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)

                out.append(
                    {
                        "prompt": user_content,
                        "chosen_actions": [p_raw],
                        "rejected_actions": [n_raw],
                        "meta": {
                            "agent_idx": current_agent_idx,
                            "step_index": depth,
                            "shared_prefix_len": depth,
                            "target_nodes": target_nodes,
                            "parent_visits": int(nd.get("visits", 0)),
                            "pos_q": float(pq),
                            "neg_q": float(nq),
                            "q_gap": float(pq - nq),
                            "final_step_only_terminal": (depth == target_nodes - 2),
                            "view": "local",
                        },
                    }
                )
                pairs_made_total += 1

    return out


def _load_trees(input_path: Path) -> Iterable[Dict[str, Any]]:
    # .json
    obj = json.loads(input_path.read_text(encoding="utf-8"))
    yield obj


def main():
    ap = argparse.ArgumentParser(
        "Preprocess MAS-MCTS trees into PRM JSONL and (optionally) PPM JSONL."
    )
    ap.add_argument(
        "--input",
        type=Path,
        default="results/mmlu_res_train_Qwen_Qwen2.5-1.5B-Instruct",
        help="Directory or file with MAS-MCTS trees",
    )
    ap.add_argument(
        "--target-nodes", type=int, default=5, help="Exact path length in nodes (root+steps)"
    )
    ap.add_argument(
        "--dedupe", action="store_true", default=True, help="Remove exact duplicate samples"
    )
    # Preference Pair Mining (PPM) switches
    ap.add_argument(
        "--make-ppm",
        action="store_true",
        help="Also emit PPM preference pairs JSONL (chosen/rejected).",
    )
    ap.add_argument(
        "--ppm-max-pairs", type=int, default=8, help="Max preference pairs to emit per tree."
    )
    ap.add_argument(
        "--ppm-pos-topk", type=int, default=4, help="Top-K positive terminal paths to consider."
    )
    ap.add_argument(
        "--ppm-neg-topk", type=int, default=4, help="Top-K negative terminal paths to consider."
    )
    ap.add_argument(
        "--ppm-root-window",
        type=float,
        nargs=2,
        default=(-0.7, 0.7),
        metavar=("LO", "HI"),
        help="Optional root-q window filter for PPM (e.g., -0.7 0.7). If omitted, no root filter is applied.",
    )
    ap.add_argument(
        "--view",
        choices=["full", "local"],
        default="full",
        help="State scope used to build training records. "
        "'full' (default) concatenates all decided steps (global view). "
        "'local' scopes each record to a single agent's inbox + output (MASPRM).",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the MAS config YAML (used only for --view local). "
        "Defaults to configs/<dataset>.yaml derived from --input's directory name.",
    )

    args = ap.parse_args()
    view_tag = "" if args.view == "full" else f"_{args.view}"
    prm_output_path = args.input.with_name(f"prm_samples{view_tag}_{args.input.stem}.jsonl")
    prm_output_path.parent.mkdir(parents=True, exist_ok=True)

    ppm_output_path = (
        args.input.with_name(f"ppm_pairs{view_tag}_{args.input.stem}.jsonl")
        if args.make_ppm
        else None
    )

    # Local view needs a MAS to replay inboxes; build it once.
    mas_local = None
    if args.view == "local":
        config_path = args.config or _derive_config_path(args.input)
        if not Path(config_path).exists():
            raise FileNotFoundError(
                f"Config YAML for local view not found: {config_path}. "
                f"Pass --config <path> to override."
            )
        print(f"[preprocess_prm][local] Using MAS config: {config_path}")
        mas_local = _build_stub_mas(Path(config_path))

    seen_prm = set()
    seen_ppm = set()
    n_files = 0

    # PRM counters (original)
    prm_n_paths = 0
    prm_n_kept = 0
    prm_n_pos = 0  # total rollouts ending with reward +1 (candidates)
    prm_n_neg = 0  # total rollouts ending with reward -1 (candidates)

    # PPM counters
    ppm_n_pairs = 0

    prm_out = prm_output_path.open("w", encoding="utf-8")
    ppm_out = ppm_output_path.open("w", encoding="utf-8") if ppm_output_path is not None else None

    json_files = sorted(args.input.rglob("*.json"))
    n_files = len(json_files)
    print(f"[preprocess_prm] Found {n_files} JSON files under {args.input.resolve()}")
    for f in tqdm(json_files, desc="Processing trees"):
        try:
            trees = list(_load_trees(f))
        except Exception:
            continue
        for t in trees:
            # PRM
            if args.view == "local":
                recs, cpos, cneg = _records_from_tree_local(
                    t,
                    mas=mas_local,
                    filename_hint=f.stem,
                    target_nodes=args.target_nodes,
                )
            else:
                recs, cpos, cneg = _records_from_tree(
                    t, filename_hint=f.stem, target_nodes=args.target_nodes
                )
            prm_n_pos += cpos
            prm_n_neg += cneg
            for r in recs:
                prm_n_paths += 1
                if args.dedupe:
                    key = (
                        r["prompt"],
                        tuple(r["completions"]),
                        tuple(round(float(x), 4) for x in r["labels"]),
                    )
                    if key in seen_prm:
                        continue
                    seen_prm.add(key)
                prm_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                prm_n_kept += 1

            # PPM
            if ppm_out is not None:
                if args.view == "local":
                    pairs = _ppm_pairs_from_tree_local(
                        t,
                        mas=mas_local,
                        filename_hint=f.stem,
                        target_nodes=args.target_nodes,
                        pos_topk=args.ppm_pos_topk,
                        neg_topk=args.ppm_neg_topk,
                        max_pairs=args.ppm_max_pairs,
                        require_root_window=(
                            tuple(args.ppm_root_window) if args.ppm_root_window else None
                        ),
                    )
                else:
                    pairs = _ppm_pairs_from_tree(
                        t,
                        filename_hint=f.stem,
                        target_nodes=args.target_nodes,
                        pos_topk=args.ppm_pos_topk,
                        neg_topk=args.ppm_neg_topk,
                        max_pairs=args.ppm_max_pairs,
                        require_root_window=(
                            tuple(args.ppm_root_window) if args.ppm_root_window else None
                        ),
                    )

                for p in pairs:
                    if args.dedupe:
                        key = tuple([p["prompt"]] + p["chosen_actions"] + p["rejected_actions"])
                        if key in seen_ppm:
                            continue
                        seen_ppm.add(key)
                    ppm_out.write(json.dumps(p, ensure_ascii=False) + "\n")
                    ppm_n_pairs += 1

    prm_out.close()
    if ppm_out is not None:
        ppm_out.close()

    print(
        f"[preprocess_prm] files={n_files} | candidates={prm_n_paths} "
        f"| R+1={prm_n_pos} | R-1={prm_n_neg} | kept={prm_n_kept} | wrote -> {prm_output_path.resolve()}"
    )
    if ppm_output_path is not None:
        print(
            f"[preprocess_prm][PPM] files={n_files}  "
            f"| pairs={ppm_n_pairs} | wrote -> {ppm_output_path.resolve()}"
        )


if __name__ == "__main__":
    main()
