# Monte Carlo Tree-of-Thoughts where each depth corresponds to a MAS agent.
# - Root depth 0 = external question
# - Depth 1..N   = agent 0..N-1
# - Leaves (after agent N-1) are terminal; we compute the final answer by
#   aggregating sink outputs (same logic as MAS.generate).
#
# Selection: UCT
# Expansion: sample multiple candidate outputs from the current agent
# Backprop:  terminal-guided (+1 if final answer matches ground truth, else -1)
#
# The exported tree.json structure is preserved:
#   {steps, action_text, is_terminal, final_answer, visits, q_sum, q_mean, children}
# Usage examples:
#   pip install "ray[default]"
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python src/run_mcts.py --dataset mmlu --split train --load_in_4bit --ray --gpus_per_actor 0.125 --actors 32

from typing import List, Dict, Any
import math
import argparse
import json
import os
from pathlib import Path
import pathlib
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging
import yaml

from mas import MAS
from agent import Agent
from mcts import Node, MAS_MCTS, is_correct
from dataset_handler import load_hard_dataset
from show_tree import build_graph, draw_tree

import ray
from tqdm.auto import tqdm


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Reduce CUDA fragmentation on long runs
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128",
)


def build_mas(
    model,
    tok,
    agent_specs: List[Dict[str, Any]],
    edges: List[List[int]],
    use_openai: bool = False,
    openai_client=None,
    openai_model: str = "gpt-4",
) -> MAS:
    """
    Build MAS from config-provided agent specs and edges.
    Each agent spec supports:
      - system_prompt (str)
      - max_new_tokens (int, default 512)
    """
    agents = []
    for spec in agent_specs:
        system_prompt = spec.get("system_prompt", "")
        max_new_tokens = int(spec.get("max_new_tokens", 512))

        agents.append(
            Agent(
                model,
                tok,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                use_openai=use_openai,
                openai_client=openai_client,
                openai_model=openai_model,
            )
        )
    return MAS(edges, agents)


def load_policy(
    model_id: str,
    device_map: str,
    load_in_4bit: bool = False,
    attn_impl: str = "sdpa",
    compile_model: bool = True,
):
    """
    Loads tokenizer and model with either fp16/fp32 or 4-bit quantization.
    device_map is now provided by argparse for flexibility.
    """
    kwargs = dict(device_map=device_map)
    kwargs["attn_implementation"] = attn_impl
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    # tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, local_files_only=True)
    # mdl = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True, **kwargs)

    mdl.eval()
    if compile_model:
        try:
            mdl = torch.compile(mdl, mode="reduce-overhead", fullgraph=False)
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}. Continuing without compile().")

    return tok, mdl


def node_to_dict(node: Node, max_children: int = 8) -> Dict[str, Any]:
    return {
        "steps": node.steps,
        "action_text": node.action_text,
        "is_terminal": node.is_terminal,
        "final_answer": node.final_answer,
        "visits": node.visits,
        "q_sum": node.q_sum,
        "q_mean": node.q_mean,
        "children": [node_to_dict(ch, max_children) for ch in node.children[:max_children]],
        "node_text": node.node_text,
    }


def print_tree(node: Node, indent: str = "", max_depth: int = 4, max_children: int = 3):
    qinfo = f"[N={node.visits}, q_sum={node.q_sum:.1f}, q_mean={node.q_mean:.3f}]"
    label = "(terminal)" if node.is_terminal else ""
    if node.action_text:
        print(f"{indent}-> {node.action_text} {label} {qinfo}")
    else:
        print(f"{indent}ROOT {qinfo}")
    if max_depth <= 0:
        return
    for ch in node.children[:max_children]:
        print_tree(ch, indent + "  ", max_depth - 1, max_children)


def _collect_terminals(node: Node) -> List[Node]:
    out = []
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur.is_terminal:
            out.append(cur)
        else:
            stack.extend(cur.children)
    return out


def _path_actions(root: Node, leaf: Node) -> List[str]:
    """Reconstruct the action_texts along the path root->leaf."""
    path, actions = [], []

    # DFS to find a path; prefer visited-heavy children for determinism
    def dfs(n, tgt, acc):
        if n is tgt:
            path.extend(acc + [n])
            return True
        for ch in sorted(n.children, key=lambda c: -c.visits):
            if dfs(ch, tgt, acc + [n]):
                return True
        return False

    dfs(root, leaf, [])
    for n in path[1:]:  # skip ROOT (no action_text)
        if n.action_text:
            actions.append(n.action_text)
    return actions


def print_top_rollouts(root: Node, truth: str, k: int = 3):
    leaves = _collect_terminals(root)
    if not leaves:
        print("\n(No terminal rollouts found.)")
        return
    # Choose top-K by visits (or use key=lambda n: n.q_mean for quality focus)
    picks = sorted(leaves, key=lambda n: (-n.visits, -n.q_mean))[:k]
    print(f"\n=== {len(picks)} rollout example(s) ===")
    for i, leaf in enumerate(picks, 1):
        score = 1 if is_correct(leaf.final_answer, truth) else -1
        actions = _path_actions(root, leaf)
        print(f"\n--- Rollout {i} ---")
        for step_idx, a in enumerate(actions, 1):
            print(f"Step {step_idx}: {a}")
        print(f"Final answer: {leaf.final_answer}")
        print(f"Reward score: {score}   (visits={leaf.visits}, q_mean={leaf.q_mean:.3f})")


@ray.remote(num_gpus=1)
class RayWorker:
    def __init__(
        self,
        model_id: str,
        device_map: str,
        load_in_4bit: bool,
        attn_impl: str,
        compile_model: bool,
        agent_specs: List[Dict[str, Any]],
        edges: List[List[int]],
        use_openai: bool = False,
        openai_api_key: str = None,
        openai_base_url: str = None,
    ):
        # Silence worker-side tqdm and noisy logs so only the driver bar is shown.
        os.environ["TQDM_DISABLE"] = "1"  # disable all tqdm in this process
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        hf_logging.set_verbosity_error()
        warnings.filterwarnings("ignore", module=".*transformers.*")

        self.use_openai = use_openai
        if self.use_openai:
            from openai import OpenAI

            self.client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
            self.openai_model = model_id

            # Load tokenizer only for Agent compatibility (optional, but good for safety)
            self.tok = AutoTokenizer.from_pretrained(
                model_id if "/" in model_id else "Qwen/Qwen2.5-1.5B-Instruct",
                use_fast=True,
                trust_remote_code=True,
            )

            self.mdl = None
        else:
            self.client = None
            self.tok, self.mdl = load_policy(
                model_id,
                device_map=device_map,
                load_in_4bit=load_in_4bit,
                attn_impl=attn_impl,
                compile_model=compile_model,
            )

        # Update build_mas call to pass new params
        self.mas = build_mas(
            self.mdl,
            self.tok,
            agent_specs,
            edges,
            use_openai=self.use_openai,
            openai_client=self.client,
            openai_model=model_id,
        )

    def run_one(
        self,
        idx: int,
        question: str,
        ground_truth_answer: str,
        c: float,
        n_candidates: int,
        gen_kwargs: Dict[str, Any],
        n_rollouts: int,
        result_path: str,
        verbose: bool,
    ):
        try:
            mcts = MAS_MCTS(
                mas=self.mas,
                question=question,
                ground_truth_answer=ground_truth_answer,
                c=c,
                n_candidates=n_candidates,
                gen_kwargs=gen_kwargs,
            )
            root = mcts.search(n_rollouts=n_rollouts)
            os.makedirs(result_path, exist_ok=True)
            json_data = node_to_dict(root)
            json_file = f"{result_path}/{idx}.json"
            png_file = f"{result_path}/{idx}.png"
            pathlib.Path(json_file).write_text(json.dumps(json_data, indent=2))
            G = build_graph(json_data, max_depth=6, max_children=4)
            draw_tree(G, png_file, dpi=400, font_size=9)
            if verbose:
                print_top_rollouts(root, ground_truth_answer, k=3)
            return {"idx": idx, "ok": True, "json": json_file, "png": png_file}
        except Exception as e:
            return {"idx": idx, "ok": False, "err": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["competition_math", "aqua_rat", "svamp", "gsm8k", "mmlu", "gpqa", "logiqa"],
        help="Name of dataset; also used to select configs/<dataset>.yaml",
    )
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")

    ap.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HF model identifier, e.g., 'Qwen/Qwen2.5-7B-Instruct'",
    )
    ap.add_argument(
        "--device_map",
        type=str,
        default="cuda",
        help="Transformers device_map, e.g., 'auto', 'cuda:0', or a JSON-like map via accelerate",
    )
    ap.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Enable 4-bit quantization (bitsandbytes).",
    )
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    ap.add_argument("--no_compile", action="store_true", help="Disable torch.compile().")

    ap.add_argument("--use_openai", action="store_true", help="Use OpenAI API for generation")
    ap.add_argument("--openai_base_url", type=str, default="https://router.huggingface.co/v1")
    ap.add_argument("--openai_api_key", type=str, default=None)

    ap.add_argument("--n_rollouts", type=int, default=64)

    # Ray parallelism flags (optional)
    ap.add_argument("--ray", action="store_true", help="Parallelize across GPUs using Ray actors.")
    ap.add_argument(
        "--actors",
        type=int,
        default=None,
        help="Number of Ray actors (model copies). Default = floor(num_gpus / gpus_per_actor), min 1.",
    )
    ap.add_argument(
        "--gpus_per_actor",
        type=float,
        default=1.0,
        help="GPU fraction per actor. Use 1.0 for one copy per GPU, 0.33 for ~3 copies per GPU, etc.",
    )

    args = ap.parse_args()

    # Load dataset
    data, q_fn, gold_fn = load_hard_dataset(args.dataset, args.split, args.n, args.seed)

    # Load MAS graph (agents + edges) from YAML
    cfg_path = Path("configs") / f"{args.dataset}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    agent_specs = cfg["agents"]
    edges = cfg["edges"]

    result_path = f"results/{args.dataset}_res_{args.split}_{args.model_id.replace('/', '_')}"

    #  Optional Ray path
    if args.ray:
        # Prepare pending jobs (skip already-finished)
        jobs = []
        for i, ex in enumerate(data):
            if Path(f"{result_path}/{i}.json").exists():
                tqdm.write(f"Skipping existing example {i}")
                continue
            q, g = q_fn(ex), gold_fn(ex)
            jobs.append((i, q, g))
        if not jobs:
            tqdm.write("Nothing to do. All examples already exist.")
            return

        # Decide number of actors
        num_gpus = max(1, torch.cuda.device_count())
        default_actors = max(1, int(math.floor(num_gpus / max(args.gpus_per_actor, 1e-6))))
        num_actors = args.actors or default_actors
        tqdm.write(f"Ray init: {num_actors} actor(s), gpus_per_actor={args.gpus_per_actor}")

        # IMPORTANT: silence worker stdout/stderr to driver
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

        # Build actors; inside, each actor loads its *own* model and MAS
        workers = [
            RayWorker.options(num_gpus=args.gpus_per_actor if not args.use_openai else 0).remote(
                args.model_id,
                device_map="cuda",
                load_in_4bit=args.load_in_4bit,
                attn_impl=args.attn_impl,
                compile_model=not args.no_compile,
                agent_specs=agent_specs,
                edges=edges,
                # Pass new args
                use_openai=args.use_openai,
                openai_api_key=args.openai_api_key,
                openai_base_url=args.openai_base_url,
            )
            for _ in range(num_actors)
        ]

        # Common run settings
        c = 8.0
        n_candidates = 4
        gen_kwargs = dict(temperature=0.7, top_p=0.95)

        # Dispatch round-robin to actors
        futures = []
        for j, (i, q, g) in enumerate(jobs):
            w = workers[j % num_actors]
            futures.append(
                w.run_one.remote(
                    i,
                    q,
                    g,
                    c,
                    n_candidates,
                    gen_kwargs,
                    args.n_rollouts,
                    result_path,
                    args.verbose,
                )
            )

        # Single global tqdm over all remaining items
        remaining = len(futures)
        in_flight = remaining
        pbar = tqdm(
            total=remaining,
            desc="All data (Ray)",
            dynamic_ncols=True,
            position=0,
            leave=False,
        )
        pbar.set_postfix_str(f"in-flight: {in_flight}")
        try:
            while futures:
                done, futures = ray.wait(futures, num_returns=1)
                res = ray.get(done[0])
                in_flight -= 1
                if res.get("ok"):
                    with pbar.external_write_mode():
                        print(
                            f"✓ idx={res['idx']}  →  "
                            f"{os.path.basename(res['json'])}, {os.path.basename(res['png'])}"
                        )
                else:
                    with pbar.external_write_mode():
                        print(f"✗ idx={res.get('idx')}  ERROR: {res.get('err')}")
                pbar.update(1)
                pbar.set_postfix_str(f"in-flight: {in_flight}")
        finally:
            pbar.close()

        ray.shutdown()
        return

    #  Single-process path
    print("Loading model...")
    if args.use_openai:
        from openai import OpenAI

        client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", use_fast=True, trust_remote_code=True
        )
        mdl = None
        print("Building MAS...")
        mas = build_mas(
            mdl,
            tok,
            agent_specs,
            edges,
            use_openai=True,
            openai_client=client,
            openai_model=args.model_id,
        )
    else:
        tok, mdl = load_policy(
            args.model_id,
            args.device_map,
            load_in_4bit=args.load_in_4bit,
            attn_impl=args.attn_impl,
            compile_model=not args.no_compile,
        )
        print("Building MAS...")
        mas = build_mas(mdl, tok, agent_specs, edges)

    print("Running MAS-MCTS...")

    remaining_idxs = [i for i in range(len(data)) if not Path(f"{result_path}/{i}.json").exists()]
    pbar_sp = tqdm(
        total=len(remaining_idxs),
        desc="All data (single proc)",
        dynamic_ncols=True,
        position=0,
        leave=False,
    )
    for i, ex in enumerate(data):
        if Path(f"{result_path}/{i}.json").exists():
            with pbar_sp.external_write_mode():
                print(f"Skipping existing example {i}")
            continue
        pbar_sp.set_postfix_str(f"idx={i}")

        question, ground_truth_answer = q_fn(ex), gold_fn(ex)
        mcts = MAS_MCTS(
            mas=mas,
            question=question,
            ground_truth_answer=ground_truth_answer,
            c=4.0,
            n_candidates=4,
            gen_kwargs=dict(temperature=0.7, top_p=0.95),
        )

        root = mcts.search(n_rollouts=args.n_rollouts)

        if args.verbose:
            with pbar_sp.external_write_mode():
                print("\n=== Top of the tree (truncated) ===")
                print_tree(root, max_depth=4, max_children=2)

        os.makedirs(result_path, exist_ok=True)
        json_data = node_to_dict(root)
        pathlib.Path(f"{result_path}/{i}.json)").write_text(json.dumps(json_data, indent=2))
        pathlib.Path(f"{result_path}/{i}.json").write_text(json.dumps(json_data, indent=2))
        G = build_graph(json_data, max_depth=6, max_children=4)
        draw_tree(G, f"{result_path}/{i}.png", dpi=400, font_size=9)
        if args.verbose:
            with pbar_sp.external_write_mode():
                print("Saved: ", f"{result_path}/{i}.json")
                print("-" * 100)
                print_top_rollouts(root, ground_truth_answer, k=3)

        pbar_sp.update(1)
    pbar_sp.close()


if __name__ == "__main__":
    main()
