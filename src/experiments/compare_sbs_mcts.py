import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import yaml
import argparse

# Project imports
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mmlu")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample_n", type=int, default=None)  # set None for full
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prm_dir",
        default="/data/milad/workspace/agentreward/checkpoints/Qwen2.5-1.5B-1.5B-PPM-qlora-512-mmlu",
    )
    parser.add_argument("--prm_base_model_id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--gen_model_id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--use_openai", action="store_true", help="Use OpenAI API for generation")

    parser.add_argument("--openai_base_url", type=str, default="https://router.huggingface.co/v1")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument(
        "--orm_dir",
        default="/data/milad/workspace/agentreward/checkpoints/Qwen2.5-1.5B-1.5B-ORM-qlora-512-mmlu",
    )
    args = parser.parse_args()

    # Dataset
    DATASET_NAME = args.dataset
    SPLIT = args.split
    SAMPLE_N = args.sample_n

    SEED = args.seed

    raw_ds, q_fn, gold_fn = load_hard_dataset(DATASET_NAME, SPLIT, n=SAMPLE_N, seed=SEED)
    data: List[Dict[str, str]] = []
    for ex in raw_ds:
        g = gold_fn(ex)
        if g is not None:
            data.append({"question": q_fn(ex), "answer": str(g)})
    print(f"[info] Loaded {len(data)} {DATASET_NAME}/{SPLIT} examples.")

    # MAS graph config

    cfg_path = Path("configs") / f"{DATASET_NAME}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    agent_specs: List[Dict[str, Any]] = cfg["agents"]
    edges: List[List[int]] = cfg["edges"]

    STEP_SEP = "</step>"

    prm_dir = args.prm_dir
    prm_base_model_id = args.prm_base_model_id
    gen_model_id = args.gen_model_id
    # Optional ORM scorer (same API as PRM loader)
    ORM_DIR = args.orm_dir or None  # allow empty string to disable

    worker_init = WorkerInit(
        agent_specs=agent_specs,
        edges=edges,
        step_separator=STEP_SEP,
        gen_model_id=gen_model_id,
        prm_dir=prm_dir,
        prm_base_model_id=prm_base_model_id,
        orm_dir=ORM_DIR,
        dtype="float16",
        seed=SEED,
        use_openai=args.use_openai,
        openai_base_url=args.openai_base_url,
        openai_api_key=args.openai_api_key,
    )

    # Conditions (descriptive, no closures)

    sbs_fast = {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 1024}
    mcts_kws = {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 1024}

    conditions: List[ConditionSpec] = [
        ConditionSpec(
            "Greedy (policy only, no PRM)", "sbs_none", {"B1": 1, "B2": 1, "gen_kwargs": sbs_fast}
        ),
        ConditionSpec(
            "Greedy + Voter (policy only, no PRM)", "voter_plain", {"k": 5, "gen_kwargs": sbs_fast}
        ),
        ConditionSpec(
            "Greedy + Voter (score-weighted, PRM-steps)",
            "voter_prm",
            {"k": 5, "gen_kwargs": sbs_fast},
        ),
        # ConditionSpec(
        #     "Greedy + Voter (score-weighted, logprob)",
        #     "voter_logprob",
        #     {"k": 5, "gen_kwargs": sbs_fast},
        # ),
        ConditionSpec(
            "SBS + PRM (B1=1, B2=5)", "sbs_prm", {"B1": 1, "B2": 5, "gen_kwargs": sbs_fast}
        ),
        ConditionSpec(
            "SBS + PRM (B1=3, B2=5)", "sbs_prm", {"B1": 3, "B2": 5, "gen_kwargs": sbs_fast}
        ),
        # ConditionSpec(
        #     "SBS + logprob (B1=1, B2=5)", "sbs_logprob", {"B1": 1, "B2": 5, "gen_kwargs": sbs_fast}
        # ),
        # ConditionSpec(
        #     "MCTS + PRM (N=16, 2 children)",
        #     "mcts_prm",
        #     {"n_simulations": 16, "max_children": 2, "c_uct": 2.0, "mcts_kwargs": mcts_kws},
        # ),
        ConditionSpec(
            "MCTS + PRM (N=10, 3 children)",
            "mcts_prm",
            {"n_simulations": 10, "max_children": 3, "c_uct": 2.0, "mcts_kwargs": mcts_kws},
        ),
        # ConditionSpec(
        #     "MCTS + logprob (N=16, 2 children)",
        #     "mcts_logprob",
        #     {"n_simulations": 16, "max_children": 2, "c_uct": 2.0, "mcts_kwargs": mcts_kws},
        # ),
    ]

    if not args.use_openai:
        conditions.append(
            ConditionSpec(
                "SBS + logprob (avg-token, B1=3, B2=5)",
                "sbs_logprob",
                {"B1": 3, "B2": 5, "gen_kwargs": sbs_fast, "logprob_agg": "avg_token"},
            )
        )

        conditions.append(
            ConditionSpec(
                "MCTS + logprob (avg-token, N=10, 3 children)",
                "mcts_logprob",
                {
                    "n_simulations": 10,
                    "max_children": 3,
                    "c_uct": 2.0,
                    "mcts_kwargs": mcts_kws,
                    "logprob_agg": "avg_token",
                },
            )
        )

        conditions.append(
            ConditionSpec(
                "Greedy + Voter (score-weighted, logprob avg-token)",
                "voter_logprob",
                {"k": 5, "gen_kwargs": sbs_fast, "logprob_agg": "avg_token"},
            )
        )

    # conditions.append(
    #     ConditionSpec(
    #         "MCTS (PUCT) + PRM (N=16, 2 children)",
    #         "mcts_prm",
    #         {
    #             "n_simulations": 16,
    #             "max_children": 2,
    #             "c_uct": 2.0,  # acts as c_puct here
    #             "uct_type": "puct",  # switch from UCT to PUCT
    #             "mcts_kwargs": mcts_kws,
    #         },
    #     )
    # )

    if ORM_DIR:
        conditions.append(
            ConditionSpec(
                "MCTS + ORM (N=10, 3 children)",
                "mcts_orm",
                {"n_simulations": 10, "max_children": 3, "c_uct": 2.0, "mcts_kwargs": mcts_kws},
            )
        )
        conditions.append(
            ConditionSpec(
                "Greedy + Voter (score-weighted, ORM end)",
                "voter_orm",
                {"k": 5, "gen_kwargs": sbs_fast},
            )
        )
        # conditions.append(
        #     ConditionSpec(
        #         "MCTS (PRM expand + ORM leaf) (N=16, 2 children)",
        #         "mcts_prm",  # we still use the 'prm' kind, but override leaf with ORM
        #         {
        #             "n_simulations": 16,
        #             "max_children": 2,
        #             "c_uct": 2.0,
        #             "mcts_kwargs": mcts_kws,
        #             "leaf_score_type": "orm",  # use ORM only at the leaf
        #         },
        #     )
        # )
        # conditions.append(
        #     ConditionSpec(
        #         "MCTS (PUCT) + ORM (N=16, 2 children)",
        #         "mcts_orm",
        #         {
        #             "n_simulations": 16,
        #             "max_children": 2,
        #             "c_uct": 2.0,
        #             "uct_type": "puct",
        #             "mcts_kwargs": mcts_kws,
        #         },
        #     )
        # )

    # Ray scaling knobs (optional)
    # For large models, safer defaults:
    os.environ.setdefault("WORKERS_PER_GPU", "1")  # 1 actor per GPU
    # os.environ.setdefault("RAY_NUM_WORKERS", "2") # Or set absolute count

    # if dataset name is different from the name in the prm dir only do MCTS + PRM also competittion means MATH
    if DATASET_NAME not in prm_dir:
        if DATASET_NAME == "competition_math":
            if "MATH" not in prm_dir:
                print(
                    f"[info] Dataset name '{DATASET_NAME}' not found in prm_dir '{prm_dir}'. "
                    "Only evaluating MCTS + PRM condition."
                )
                conditions = [cond for cond in conditions if cond.name.startswith("MCTS + PRM")]
        else:
            print(
                f"[info] Dataset name '{DATASET_NAME}' not found in prm_dir '{prm_dir}'. "
                "Only evaluating MCTS + PRM condition."
            )
            conditions = [cond for cond in conditions if cond.name.startswith("MCTS + PRM")]

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


if __name__ == "__main__":
    main()
