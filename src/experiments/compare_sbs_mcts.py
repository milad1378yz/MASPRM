import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset_handler import load_hard_dataset

from ray_eval import (
    WorkerInit,
    ConditionSpec,
    evaluate_conditions_ray,
)

os.environ.setdefault(
    "PYTORCH_ALLOC_CONF",
    "max_split_size_mb:128",
)

METHODS = (
    "single_pass",
    "sbs_prm",
    "sbs_logprob",
    "mcts_prm",
    "mcts_logprob",
    "mcts_orm",
    "voter_plain",
    "voter_prm",
    "voter_logprob",
    "voter_orm",
    "handoff_fixed",
    "handoff_random",
    "handoff_policy",
    "handoff_prm",
)
DEFAULT_METHODS = ("single_pass", "mcts_prm")


def build_condition_specs(args: argparse.Namespace) -> List[ConditionSpec]:
    sbs_kwargs = {
        "temperature": args.sbs_temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    mcts_kwargs = {
        "temperature": args.mcts_temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    conditions: List[ConditionSpec] = []

    for method in dict.fromkeys(args.methods):
        if method.startswith("handoff_"):
            handoff_names = {
                "handoff_fixed": "Fixed schedule + policy likelihood",
                "handoff_random": "Random handoff + policy likelihood",
                "handoff_policy": "Agent-selected handoff + policy likelihood",
                "handoff_prm": "Agent-selected handoff + MASPRM",
            }
            conditions.append(
                ConditionSpec(
                    handoff_names[method],
                    method,
                    {
                        "handoff_candidates": args.handoff_candidates,
                        "min_turns": args.min_turns,
                        "max_turns": args.max_turns,
                        "gen_kwargs": sbs_kwargs,
                        "logprob_agg": args.handoff_logprob_agg,
                        "prm_context_length": args.prm_context_length,
                        "allow_self": args.allow_self_handoff,
                        "random_can_finalize": not args.random_fixed_horizon,
                    },
                )
            )
            continue

        if method == "single_pass":
            conditions.append(
                ConditionSpec(
                    "DyLAN single pass (no scorer)",
                    "sbs_none",
                    {"B1": 1, "B2": 1, "gen_kwargs": sbs_kwargs},
                )
            )
            continue

        family, scorer = method.split("_", 1)
        if family == "sbs":
            params = {"B1": args.b1, "B2": args.b2, "gen_kwargs": sbs_kwargs}
            details = [f"B1={args.b1}", f"B2={args.b2}"]
            name = f"SBS + {'PRM' if scorer == 'prm' else 'logprob'}"
        elif family == "mcts":
            params = {
                "n_simulations": args.n_simulations,
                "max_children": args.max_children,
                "c_uct": args.c_uct,
                "mcts_kwargs": mcts_kwargs,
            }
            details = [f"N={args.n_simulations}", f"{args.max_children} children"]
            search_name = "MCTS" if args.uct_type == "uct" else "MCTS (PUCT)"
            scorer_name = {
                "prm": "PPM",
                "logprob": "logprob",
                "orm": "ORM",
            }[scorer]
            name = f"{'DyLAN ' if method == 'mcts_prm' else ''}{search_name} + {scorer_name}"
            if args.uct_type != "uct":
                params["uct_type"] = args.uct_type
            if method == "mcts_prm" and args.orm_leaf:
                params["leaf_score_type"] = "orm"
                details.insert(0, "ORM leaf")
        else:
            params = {"k": args.voter_k, "gen_kwargs": sbs_kwargs}
            details = []
            voter_descriptions = {
                "plain": "policy only, no PRM",
                "prm": "score-weighted, PRM-steps",
                "logprob": "score-weighted, logprob",
                "orm": "score-weighted, ORM end",
            }
            name = f"Greedy + Voter ({voter_descriptions[scorer]})"

        if scorer in {"prm", "orm"}:
            params["view_mode"] = args.view_mode
        if scorer == "logprob" and args.logprob_agg != "sum":
            params["logprob_agg"] = args.logprob_agg
            details.insert(0, args.logprob_agg.replace("_", "-"))
        if details:
            name += f" ({', '.join(details)})"

        conditions.append(ConditionSpec(name, method, params))

    return conditions


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_n", type=int, default=None)  # set None for full
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prm_dir",
        default="checkpoints/Qwen2.5-1.5B-RM",
    )
    parser.add_argument("--prm_base_model_id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--handoff_prm_dir",
        default=None,
        help="Explicit frozen MASPRM checkpoint for handoff_prm (required for that row).",
    )
    parser.add_argument(
        "--handoff_prm_base_model_id",
        default=None,
        help="Optional base model for --handoff_prm_dir; otherwise read from its adapter config.",
    )
    parser.add_argument("--gen_model_id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--view_mode",
        choices=["full", "local"],
        default="full",
        help="State scope for PRM/ORM scoring: routed transcript ('full') or agent-local view.",
    )
    parser.add_argument(
        "--orm_dir",
        default="",
    )
    parser.add_argument(
        "--mas_config",
        default=None,
        help="Optional path to a MAS YAML config. Defaults to configs/<dataset>.yaml.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(DEFAULT_METHODS),
        metavar="METHOD",
        help=(
            "Methods to evaluate in the given order. "
            f"Defaults to: {' '.join(DEFAULT_METHODS)}. "
            f"Available: {', '.join(METHODS)}."
        ),
    )
    method_args = parser.add_argument_group("method hyperparameters")
    method_args.add_argument("--b1", type=int, default=3, help="SBS retained beam width.")
    method_args.add_argument(
        "--b2", type=int, default=5, help="SBS candidates sampled per expansion."
    )
    method_args.add_argument(
        "--voter_k", type=int, default=5, help="Number of voter candidates."
    )
    method_args.add_argument(
        "--n_simulations", type=int, default=10, help="MCTS simulation budget."
    )
    method_args.add_argument(
        "--max_children", type=int, default=3, help="Maximum MCTS children per node."
    )
    method_args.add_argument("--c_uct", type=float, default=2.0)
    method_args.add_argument("--uct_type", choices=["uct", "puct"], default="uct")
    method_args.add_argument(
        "--orm_leaf",
        action="store_true",
        help="Use ORM leaf scoring with mcts_prm.",
    )
    method_args.add_argument(
        "--logprob_agg",
        choices=["sum", "avg_token", "avg_step"],
        default="sum",
    )
    method_args.add_argument("--sbs_temperature", type=float, default=1.0)
    method_args.add_argument("--mcts_temperature", type=float, default=0.6)
    method_args.add_argument("--top_p", type=float, default=0.95)
    method_args.add_argument("--max_new_tokens", type=int, default=1024)
    handoff_args = parser.add_argument_group("dynamic handoff hyperparameters")
    handoff_args.add_argument("--handoff_candidates", type=int, default=3)
    handoff_args.add_argument("--min_turns", type=int, default=4)
    handoff_args.add_argument("--max_turns", type=int, default=12)
    handoff_args.add_argument(
        "--handoff_logprob_agg",
        choices=["sum", "avg_token"],
        default="avg_token",
    )
    handoff_args.add_argument("--prm_context_length", type=int, default=16384)
    handoff_args.add_argument("--allow_self_handoff", action="store_true")
    handoff_args.add_argument(
        "--random_fixed_horizon",
        action="store_true",
        help="Prevent the random-routing control from sampling FINAL before max_turns.",
    )
    handoff_args.add_argument("--math_levels", nargs="+", type=int, default=[4, 5])
    handoff_args.add_argument("--math_stratify_by", default="type")
    args = parser.parse_args()

    uses_handoff = any(method.startswith("handoff_") for method in args.methods)
    if uses_handoff and any(
        not method.startswith("handoff_") for method in args.methods
    ):
        parser.error(
            "handoff and static methods require different MAS configs; run them in "
            "separate invocations"
        )
    if args.dataset is None:
        args.dataset = "competition_math" if uses_handoff else "mmlu"
    if uses_handoff and args.sample_n is None:
        args.sample_n = 200

    for arg_name in (
        "b1",
        "b2",
        "voter_k",
        "n_simulations",
        "max_children",
        "max_new_tokens",
        "handoff_candidates",
        "min_turns",
        "max_turns",
        "prm_context_length",
    ):
        if getattr(args, arg_name) < 1:
            parser.error(f"--{arg_name} must be at least 1")
    if args.c_uct < 0:
        parser.error("--c_uct cannot be negative")
    if args.sbs_temperature <= 0 or args.mcts_temperature <= 0:
        parser.error("temperatures must be greater than zero")
    if not 0 < args.top_p <= 1:
        parser.error("--top_p must be in the interval (0, 1]")
    if args.orm_leaf and "mcts_prm" not in args.methods:
        parser.error("--orm_leaf requires the mcts_prm method")
    if args.max_turns < args.min_turns:
        parser.error("--max_turns must be greater than or equal to --min_turns")

    conditions = build_condition_specs(args)

    needs_orm = any(
        condition.kind in {"mcts_orm", "voter_orm"}
        or condition.params.get("leaf_score_type") == "orm"
        for condition in conditions
    )
    if "handoff_prm" in args.methods and not args.handoff_prm_dir:
        parser.error(
            "--handoff_prm_dir is required for handoff_prm; the requested MASPRM-7B "
            "checkpoint is not available at the repository's 1.5B default"
        )
    static_prm_methods = {"sbs_prm", "mcts_prm", "voter_prm"}
    if any(method in static_prm_methods for method in args.methods) and not args.prm_dir:
        parser.error("--prm_dir is required by the selected static PRM methods")
    if needs_orm and not args.orm_dir:
        parser.error("--orm_dir is required by the selected methods")

    # Dataset
    DATASET_NAME = args.dataset
    SPLIT = args.split
    SAMPLE_N = args.sample_n

    SEED = args.seed

    raw_ds, q_fn, gold_fn = load_hard_dataset(
        DATASET_NAME,
        SPLIT,
        n=SAMPLE_N,
        seed=SEED,
        levels=(args.math_levels if uses_handoff and DATASET_NAME == "competition_math" else None),
        stratify_by=(
            args.math_stratify_by
            if uses_handoff and DATASET_NAME == "competition_math"
            else None
        ),
    )
    data: List[Dict[str, str]] = []
    for ex in raw_ds:
        g = gold_fn(ex)
        if g is not None:
            data.append({"question": q_fn(ex), "answer": str(g)})
    print(f"[info] Loaded {len(data)} {DATASET_NAME}/{SPLIT} examples.")
    if uses_handoff and len(data) != SAMPLE_N:
        parser.error(
            f"requested {SAMPLE_N} handoff examples but only {len(data)} have answers "
            "supported by the current evaluator; choose a symbolic MATH verifier or "
            "an explicitly numeric-only stratified subset before running this experiment"
        )

    # MAS graph config

    cfg_path = (
        Path(args.mas_config)
        if args.mas_config
        else (
            Path("configs/math_handoff8.yaml")
            if uses_handoff
            else Path("configs") / f"{DATASET_NAME}.yaml"
        )
    )
    cfg = yaml.safe_load(cfg_path.read_text())
    agent_specs: List[Dict[str, Any]] = cfg["agents"]
    edges: List[List[int]] = cfg.get("edges", [])
    handoff_config = dict(cfg.get("handoff", {})) if uses_handoff else None
    if uses_handoff and not handoff_config:
        parser.error("Handoff methods require a top-level 'handoff' section in --mas_config")
    if handoff_config:
        for condition in conditions:
            if condition.kind.startswith("handoff_"):
                condition.params.setdefault("initial_speaker", handoff_config.get("initial_role"))
                condition.params.setdefault("fixed_schedule", handoff_config.get("fixed_schedule"))

    STEP_SEP = "</step>"

    selected_prm_dir = (
        args.handoff_prm_dir if "handoff_prm" in args.methods else args.prm_dir
    )
    prm_dir = (
        str(Path(selected_prm_dir).resolve()) if selected_prm_dir else selected_prm_dir
    )
    prm_base_model_id = (
        args.handoff_prm_base_model_id
        if "handoff_prm" in args.methods
        else args.prm_base_model_id
    )
    gen_model_id = args.gen_model_id
    # Optional ORM scorer (same API as PRM loader)
    ORM_DIR = (
        str(Path(args.orm_dir).resolve()) if args.orm_dir else None
    )  # allow empty string to disable

    worker_init = WorkerInit(
        agent_specs=agent_specs,
        edges=edges,
        handoff_config=handoff_config,
        step_separator=STEP_SEP,
        gen_model_id=gen_model_id,
        prm_dir=prm_dir,
        prm_base_model_id=prm_base_model_id,
        orm_dir=ORM_DIR,
        prm_max_length=2048,
        dtype="float16",
        seed=SEED,
    )

    # Ray scaling knobs (optional)
    # For large models, safer defaults:
    os.environ.setdefault("WORKERS_PER_GPU", "1")  # 1 actor per GPU
    # os.environ.setdefault("RAY_NUM_WORKERS", "2") # Or set absolute count

    # Evaluate (models are constructed inside workers)
    results = evaluate_conditions_ray(
        data=data,
        worker_init=worker_init,
        conditions=conditions,
        name_dataset=DATASET_NAME,
    )

    if not results:
        print("No results obtained.")
        return
    # Pretty print table
    print("\n" + "#" * 80)
    print("Table-4 style results")
    print("#" * 80)
    name_width = max(len(k) for k in results) + 2
    max_k = 5
    header_metrics = "  ".join([f"p@{i}" for i in range(1, max_k + 1)])
    print(f"{'Condition':<{name_width}}  {header_metrics}   {'Tokens (avg±std)':>22}")
    print("-" * (name_width + 32 + 10 * max_k))
    for cond_name, v in results.items():
        # pass_at_1 is identical to accuracy
        passes = [
            v.get(f"pass_at_{i}", v.get("accuracy", 0.0 if i == 1 else 0.0))
            for i in range(1, max_k + 1)
        ]
        tmean = v.get("tok_mean", 0.0)
        tstd = v.get("tok_std", 0.0)
        pass_cols = "  ".join(f"{p:>6.3f}" for p in passes)
        print(f"{cond_name:<{name_width}}  {pass_cols}   {tmean:>8.1f}±{tstd:<8.1f}")
    print("#" * 80)

    handoff_results = {
        name: values for name, values in results.items() if "hit_at_4" in values
    }
    if handoff_results:
        print("\nDynamic-handoff prefix results")
        print(f"{'Condition':<{name_width}}  {'Hit@4':>7}  {'Hit@8':>7}  {'Hit@12':>7}  {'Depth':>7}")
        for cond_name, values in handoff_results.items():
            print(
                f"{cond_name:<{name_width}}  "
                f"{values['hit_at_4']:>7.3f}  {values['hit_at_8']:>7.3f}  "
                f"{values['hit_at_12']:>7.3f}  "
                f"{values['realized_depth_mean']:>7.2f}"
            )
        comparison = next(
            (
                values
                for values in handoff_results.values()
                if "paired_accuracy_delta" in values
            ),
            None,
        )
        if comparison:
            for threshold in (8, 12):
                prefix = f"t{threshold}_"
                print(
                    f"T={threshold} dynamic MASPRM − policy: "
                    f"Δ={comparison[prefix + 'paired_accuracy_delta']:.3f}, "
                    f"95% CI [{comparison[prefix + 'paired_bootstrap_ci_low']:.3f}, "
                    f"{comparison[prefix + 'paired_bootstrap_ci_high']:.3f}], "
                    f"McNemar p={comparison[prefix + 'mcnemar_exact_p']:.4g}"
                )


if __name__ == "__main__":
    main()
