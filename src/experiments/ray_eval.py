from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Dict, List, Any
import hashlib
import json
import math
import os
import queue
import statistics
import time
from pathlib import Path

import torch
from tqdm import tqdm
import ray
from ray.util.queue import Queue

from answer_utils import is_correct
from handoff_mas import build_handoff_mas_from_specs
from mas import MAS, build_mas_from_specs


from core import (
    _seed_everything,
    TokenStats,
    handoff_sbs_decode,
    sbs_decode2,
    make_scored_voter,
    make_llm_action_ranker,
    load_prm_scorer,
    ensure_separator_token,
    align_model_to_tokenizer,
    build_runtime,
    MCTSInfer,
    _majority_by_numbers_equal,
)

HANDOFF_KINDS = {
    "handoff_fixed",
    "handoff_random",
    "handoff_policy",
    "handoff_prm",
}


@dataclass
class WorkerInit:
    # MAS graph
    agent_specs: List[Dict[str, Any]]
    edges: List[List[int]]
    handoff_config: Optional[Dict[str, Any]] = None
    step_separator: str = "</step>"

    # Models
    gen_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    prm_dir: Optional[str] = None
    prm_base_model_id: Optional[str] = None
    orm_dir: Optional[str] = None
    judge_model_id: Optional[str] = None
    judge_load_in_4bit: bool = False
    prm_max_length: int = 2048
    attn_impl: str = "sdpa"

    # dtype options: "float16" | "bfloat16" | "float32"
    dtype: str = "float16"
    seed: int = 0


@dataclass
class ConditionSpec:
    name: str
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").lower()
    if s in ("float16", "fp16", "half"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float32", "fp32", "full", "float"):
        return torch.float32
    return torch.float16


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_identity(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    checkpoint = Path(path).resolve()
    identity: Dict[str, Any] = {"path": str(checkpoint), "exists": checkpoint.exists()}
    if checkpoint.is_dir():
        identity["files"] = [
            {
                "path": str(file.relative_to(checkpoint)),
                "size": file.stat().st_size,
                "mtime_ns": file.stat().st_mtime_ns,
            }
            for file in sorted(checkpoint.rglob("*"))
            if file.is_file()
        ]
    elif checkpoint.is_file():
        stat = checkpoint.stat()
        identity.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return identity


def _completed_cache_is_valid(
    results_path: str,
    summary_path: str,
    records_path: str,
    *,
    manifest_hash: str,
    expected_count: int,
) -> bool:
    """Require a complete, manifest-matched record set before skipping a run."""
    try:
        result_lines = Path(results_path).read_text().splitlines()
        summary = json.loads(Path(summary_path).read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(summary, dict) or not any(
        line.strip() == "COMPLETED" for line in result_lines
    ):
        return False

    seen = set()
    try:
        with Path(records_path).open() as records_file:
            for line in records_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                idx = int(record["index"])
                if (
                    idx in seen
                    or idx < 0
                    or idx >= int(expected_count)
                    or record.get("manifest_hash") != manifest_hash
                ):
                    return False
                seen.add(idx)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False
    return seen == set(range(int(expected_count)))


def evaluate_conditions_ray(
    data: List[Dict[str, str]],
    worker_init: WorkerInit,
    conditions: List[ConditionSpec],
    name_dataset: str,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate conditions with Ray workers. Workers are created per condition so
    each pool only loads the models that condition actually needs.

    Returns: {name: {"accuracy": float, "tok_mean": float, "tok_std": float}}
    """
    # We will compute pass@k for these k values (k is over *independent runs* of the method)
    MAX_PASS_K = 5
    REPORTED_PASS_K = (1, 3, 5)

    # --- Ray init (idempotent)
    if not ray.is_initialized():
        # Best-effort working_dir/PYTHONPATH for cluster workers
        src_dir = str(Path(__file__).resolve().parent.parent)  # .../src
        existing = os.environ.get("PYTHONPATH", "")

        env_vars_to_pass = {
            "PYTHONPATH": f"{existing}:{src_dir}" if existing else src_dir,
            # Pass the cache location to the workers
            "HF_HOME": os.environ.get("HF_HOME", "/tmp/hf"),
            # Pass the offline mode setting
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "0"),
        }
        hf_hub_cache = os.environ.get("HF_HUB_CACHE")
        if hf_hub_cache:
            env_vars_to_pass["HF_HUB_CACHE"] = hf_hub_cache

        runtime_env = {
            "env_vars": env_vars_to_pass,
            "working_dir": str(Path(__file__).resolve().parents[2]),  # repo root
        }
        ray.init(ignore_reinit_error=True, log_to_driver=True, runtime_env=runtime_env)

    cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
    workers_per_gpu = max(1, int(os.environ.get("WORKERS_PER_GPU", "1")))
    override = os.environ.get("RAY_NUM_WORKERS", None)

    if override:
        num_workers = int(override)
        per_worker_gpu = (1.0 * cluster_gpus / num_workers) if cluster_gpus > 0 else 0.0
    else:
        if cluster_gpus > 0:
            num_workers = max(1, cluster_gpus * workers_per_gpu)
            per_worker_gpu = 1.0 / workers_per_gpu
        else:
            num_workers = os.cpu_count() or 1
            per_worker_gpu = 0.0

    # --- Worker actor --
    @ray.remote(num_cpus=1, num_gpus=per_worker_gpu)
    class EvalWorker:
        def __init__(self, init: WorkerInit, dtype: str, rank: int):
            self.rank = int(rank)
            worker_seed = int((init.seed or 0) + 1009 * self.rank + 17)
            _seed_everything(worker_seed)
            self.step_sep = init.step_separator
            self.agent_specs = init.agent_specs
            self.edges = init.edges
            self.handoff_config = dict(init.handoff_config or {})
            self.dtype = _dtype_from_str(dtype)

            # 1) Load PRM scorer (optional)
            self.prm_score_fn = None
            self.prm_tok = None
            self.prm_model = None
            if init.prm_dir:
                self.prm_score_fn, self.prm_tok, self.prm_model = load_prm_scorer(
                    init.prm_dir,
                    base_model_id=init.prm_base_model_id,
                    max_length=int(init.prm_max_length),
                    torch_dtype=self.dtype,
                    step_separator=self.step_sep,
                    attn_impl=init.attn_impl,
                )

                # modify the PRM tokenizer/model with the separator
                added = ensure_separator_token(self.prm_tok, self.step_sep)
                print(f"[Worker {self.rank}] Added {added} special tokens for step separator.")
                if added > 0 and self.prm_model is not None:
                    align_model_to_tokenizer(self.prm_model, self.prm_tok)
            # 2) Load the generation runtime.
            self.runtime = build_runtime(
                init.gen_model_id,
                # Handoff workers are reused across all four rows, so the
                # policy LM must not change tokenizer when the PRM row is present.
                tokenizer=None if self.handoff_config else self.prm_tok,
                torch_dtype=self.dtype,
                attn_impl=init.attn_impl,
            )
            self.handoff_mas = (
                build_handoff_mas_from_specs(
                    self.runtime.model,
                    self.runtime.tokenizer,
                    self.agent_specs,
                )
                if self.handoff_config
                else None
            )

            # 3) Optional zero-shot judge for stepwise action ranking.
            self.judge_ranker = None
            if init.judge_model_id:
                judge_runtime = build_runtime(
                    init.judge_model_id,
                    tokenizer=None,
                    torch_dtype=self.dtype,
                    load_in_4bit=bool(init.judge_load_in_4bit),
                    attn_impl=init.attn_impl,
                )
                self.judge_ranker = make_llm_action_ranker(
                    judge_runtime.model, judge_runtime.tokenizer
                )

            # 4) ORM scorer (optional) — same API as PRM loader
            self.orm_score_fn = None
            self.orm_tok = None
            self.orm_model = None
            if init.orm_dir:
                self.orm_score_fn, self.orm_tok, self.orm_model = load_prm_scorer(
                    init.orm_dir,
                    base_model_id=None,  # read from adapter config
                    torch_dtype=self.dtype,
                    step_separator=self.step_sep,
                    attn_impl=init.attn_impl,
                )
                added = ensure_separator_token(self.orm_tok, self.step_sep)
                print(f"[Worker {self.rank}] Added {added} special tokens for ORM separator.")
                if added > 0 and self.orm_model is not None:
                    align_model_to_tokenizer(self.orm_model, self.orm_tok)

        # --- decoders by spec.kind
        def _decode_by_spec(
            self,
            q: str,
            spec: ConditionSpec,
            example_seed: Optional[int] = None,
        ):
            kind = spec.kind
            p = spec.params or {}

            # Defaults
            sbs_kwargs = p.get(
                "gen_kwargs",
                {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 1024},
            )
            mcts_kwargs = p.get(
                "mcts_kwargs",
                {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 1024},
            )
            B1 = int(p.get("B1", 1))
            B2 = int(p.get("B2", 5))
            n_sims = int(p.get("n_simulations", 40))
            max_children = int(p.get("max_children", 3))
            c_uct = float(p.get("c_uct", 2.0))
            k = int(p.get("k", 5))

            pass_k = max(1, int(p.get("pass_k", 5)))

            # Scope of state text fed to PRM/ORM: "full" = routed transcript,
            # "local" = only what the scored agent itself received + produced.
            view_mode = str(p.get("view_mode", "full")).lower()

            if kind in HANDOFF_KINDS:
                if self.handoff_mas is None:
                    raise ValueError(
                        "A handoff condition requires a config with a handoff section."
                    )
                route_mode = {
                    "handoff_fixed": "fixed",
                    "handoff_random": "random",
                    "handoff_policy": "agent",
                    "handoff_prm": "agent",
                }[kind]
                score_type = "prm" if kind == "handoff_prm" else "logprob"
                result = handoff_sbs_decode(
                    self.handoff_mas,
                    q,
                    route_mode=route_mode,
                    score_type=score_type,
                    score_fn=self.prm_score_fn if score_type == "prm" else None,
                    prm_tokenizer=self.prm_tok if score_type == "prm" else None,
                    n_candidates=int(p.get("handoff_candidates", 3)),
                    min_turns=int(p.get("min_turns", 4)),
                    max_turns=int(p.get("max_turns", 12)),
                    initial_speaker=str(
                        p.get(
                            "initial_speaker",
                            self.handoff_config.get("initial_role", "Problem Analyst"),
                        )
                    ),
                    fixed_schedule=p.get(
                        "fixed_schedule",
                        self.handoff_config.get("fixed_schedule"),
                    ),
                    gen_kwargs=sbs_kwargs,
                    step_separator=self.step_sep,
                    max_context_tokens=int(p.get("prm_context_length", 16384)),
                    seed=int(example_seed if example_seed is not None else p.get("seed", 42)),
                    allow_self=bool(p.get("allow_self", False)),
                    random_can_finalize=bool(p.get("random_can_finalize", True)),
                )
                metrics = result.metrics()
                return (
                    result.answer,
                    [result.answer],
                    result.usage.generated,
                    result.usage.prm_calls,
                    result.usage.agent_runs,
                    metrics,
                )

            build_mas = partial(
                build_mas_from_specs,
                self.runtime.model,
                self.runtime.tokenizer,
                self.agent_specs,
                self.edges,
            )

            # Helper: single-run decode with trace (policy-only)
            def base_policy_with_trace(mas: MAS, q: str):
                pred, usage, tr = sbs_decode2(
                    mas,
                    q,
                    score_type="none",
                    B1=1,
                    B2=1,
                    gen_kwargs=sbs_kwargs,
                    step_separator=self.step_sep,
                    return_trace=True,
                )
                return pred, usage, tr

            # Build MAS for the first run
            mas0 = build_mas()

            #  SBS variants
            if kind == "sbs_none":
                finals: List[str] = []
                pred, usage, trace = sbs_decode2(
                    mas0,
                    q,
                    score_type="none",
                    B1=B1,
                    B2=B2,
                    gen_kwargs=sbs_kwargs,
                    step_separator=self.step_sep,
                    final_answers_out=finals,
                    return_trace=True,
                )
                candidates = finals[:pass_k] if finals else [pred]
                agent_traces = [
                    {
                        "agent_idx": int(step["agent_idx"]),
                        "agent_name": mas0.agents[int(step["agent_idx"])].name,
                        "trace": str(step["chosen_text"]),
                    }
                    for step in trace
                    if step.get("chosen_text")
                ]
                tool_traces = [
                    step
                    for step in agent_traces
                    if mas0.agents[step["agent_idx"]].python_tool is not None
                ]
                agent_stats = [
                    {
                        "agent_idx": idx,
                        "agent_name": agent.name,
                        "model_responses": agent.last_tool_stats.model_responses,
                        "max_prompt_tokens": agent.last_tool_stats.max_prompt_tokens,
                        "truncated_responses": agent.last_tool_stats.truncated_responses,
                    }
                    for idx, agent in enumerate(mas0.agents)
                ]
                return (
                    pred,
                    candidates,
                    usage.generated,
                    usage.prm_calls,
                    usage.agent_runs,
                    {
                        "tool_usage": usage.tool_usage(),
                        "agent_traces": agent_traces,
                        "tool_traces": tool_traces,
                        "agent_stats": agent_stats,
                    },
                )

            if kind == "sbs_prm":
                finals: List[str] = []
                pred, usage = sbs_decode2(
                    mas0,
                    q,
                    score_type="prm",
                    score_fn=self.prm_score_fn,
                    prm_tokenizer=self.prm_tok,
                    B1=B1,
                    B2=B2,
                    gen_kwargs=sbs_kwargs,
                    step_separator=self.step_sep,
                    final_answers_out=finals,
                    view_mode=view_mode,
                )
                candidates = finals[:pass_k] if finals else [pred]
                return (
                    pred,
                    candidates,
                    usage.generated,
                    usage.prm_calls,
                    usage.agent_runs,
                )

            if kind == "sbs_logprob":
                finals: List[str] = []
                pred, usage = sbs_decode2(
                    mas0,
                    q,
                    score_type="logprob",
                    B1=B1,
                    B2=B2,
                    gen_kwargs=sbs_kwargs,
                    step_separator=self.step_sep,
                    final_answers_out=finals,
                )
                candidates = finals[:pass_k] if finals else [pred]
                return (
                    pred,
                    candidates,
                    usage.generated,
                    usage.prm_calls,
                    usage.agent_runs,
                )

            #  MCTS variants
            if kind in ("mcts_prm", "mcts_logprob", "mcts_orm", "mcts_judge"):
                score_type = {
                    "mcts_prm": "prm",
                    "mcts_logprob": "logprob",
                    "mcts_orm": "orm",
                    "mcts_judge": "judge",
                }[kind]
                if score_type == "judge" and self.judge_ranker is None:
                    raise ValueError("mcts_judge requires judge_model_id.")
                score_fn = (
                    self.prm_score_fn
                    if score_type == "prm"
                    else (self.orm_score_fn if score_type == "orm" else None)
                )
                prm_tok = self.prm_tok if score_type == "prm" else None
                orm_tok = self.orm_tok if score_type == "orm" else None

                infer = MCTSInfer(
                    mas0,
                    q,
                    score_fn,
                    score_type=score_type,
                    prm_tokenizer=prm_tok,
                    orm_tokenizer=orm_tok,
                    max_children=max_children,
                    c_uct=c_uct,
                    gen_kwargs=mcts_kwargs,
                    step_separator=self.step_sep,
                    view_mode=view_mode,
                    action_ranker=(self.judge_ranker if score_type == "judge" else None),
                )
                start_time = time.perf_counter()
                pred, usage = infer.decode(n_simulations=n_sims)
                latency = time.perf_counter() - start_time
                try:
                    candidates = infer.get_topk_answers(k=pass_k)
                except Exception:
                    candidates = []
                if not candidates:
                    candidates = [pred]
                metadata: Dict[str, Any] = {}
                if infer.judge_calls:
                    metadata = {
                        "judge_calls": int(infer.judge_calls),
                        "judge_parse_failures": int(infer.judge_parse_failures),
                        "latency_s": float(latency),
                    }
                return (
                    pred,
                    candidates,
                    usage.generated,
                    usage.prm_calls,
                    usage.agent_runs,
                    metadata,
                )

            #  Majority voters
            if kind == "voter_plain":
                preds: List[str] = []
                total = TokenStats()
                p0, u0, _ = base_policy_with_trace(mas0, q)
                preds.append(p0)
                total.add(u0)
                for _ in range(max(0, k - 1)):
                    mas_i = build_mas()
                    pi, ui, _ = base_policy_with_trace(mas_i, q)
                    preds.append(pi)
                    total.add(ui)
                pred = _majority_by_numbers_equal(preds)
                candidates = preds[:pass_k] if pass_k > 0 else [pred]
                return (
                    pred,
                    candidates,
                    total.generated,
                    total.prm_calls,
                    total.agent_runs,
                )

            if kind in ("voter_prm", "voter_logprob", "voter_orm"):
                # Build make_scored_voter *inside* the actor (no cross-process capture)
                score_mode = {
                    "voter_prm": "prm",
                    "voter_logprob": "logprob",
                    "voter_orm": "orm",
                }[kind]

                def base_with_trace(mas: MAS, q: str):
                    return sbs_decode2(
                        mas,
                        q,
                        score_type="none",
                        B1=1,
                        B2=1,
                        gen_kwargs=sbs_kwargs,
                        step_separator=self.step_sep,
                        return_trace=True,
                    )

                voter = make_scored_voter(
                    base_with_trace,
                    build_mas,
                    score_mode=score_mode,
                    score_fn=(
                        self.prm_score_fn
                        if score_mode == "prm"
                        else (self.orm_score_fn if score_mode == "orm" else None)
                    ),
                    prm_tokenizer=(self.prm_tok if score_mode == "prm" else None),
                    orm_tokenizer=self.orm_tok if score_mode == "orm" else None,
                    step_separator=self.step_sep,
                    k=k,
                    return_all_answers=True,
                    view_mode=view_mode,
                )
                res = voter(mas0, q)
                if isinstance(res, tuple) and len(res) == 3:
                    pred, usage, raw_answers = res
                    candidates = (
                        list(raw_answers[:pass_k]) if isinstance(raw_answers, list) else [pred]
                    )
                elif isinstance(res, tuple) and len(res) == 2:
                    pred, usage = res
                    candidates = [pred]
                else:
                    pred = str(res)
                    usage = TokenStats()
                    candidates = [pred]
                return (
                    pred,
                    candidates,
                    usage.generated,
                    usage.prm_calls,
                    usage.agent_runs,
                )

            raise ValueError(f"Unknown evaluation condition: {kind!r}.")

        # --- queue consumer
        def consume(self, spec: ConditionSpec, in_q: "Queue", out_q: "Queue"):
            try:
                while True:
                    item = in_q.get()
                    if item is None:
                        break
                    idx, q, gold, ex_seed = item
                    _seed_everything(int(ex_seed))
                    metadata: Dict[str, Any] = {}
                    try:
                        res = self._decode_by_spec(q, spec, example_seed=int(ex_seed))
                        # normalize length (be paranoid)
                        if isinstance(res, tuple):
                            if len(res) == 6:
                                (
                                    pred,
                                    cand_list,
                                    tok_total,
                                    prm_calls,
                                    agent_runs,
                                    metadata,
                                ) = res
                            elif len(res) == 5:
                                pred, cand_list, tok_total, prm_calls, agent_runs = res
                            elif len(res) == 4:
                                pred, tok_total, prm_calls, agent_runs = res
                                cand_list = [pred]
                            elif len(res) == 2:
                                pred, tok_total = res
                                cand_list = [pred]
                                prm_calls = 0
                                agent_runs = 0
                            else:
                                pred = str(res[0])
                                cand_list = [pred]
                                tok_total = 0
                                prm_calls = 0
                                agent_runs = 0
                        else:
                            pred = str(res)
                            cand_list = [pred]
                            tok_total = 0
                            prm_calls = 0
                            agent_runs = 0
                    except Exception as e:
                        # surface the error as a record instead of killing the actor
                        pred = f"[ERROR] {e}"
                        cand_list = [pred]
                        tok_total = 0
                        prm_calls = 0
                        agent_runs = 0
                        metadata = {"error": str(e)}
                    out_q.put(
                        (
                            idx,
                            pred,
                            cand_list,
                            tok_total,
                            prm_calls,
                            agent_runs,
                            metadata,
                            gold,
                        )
                    )
            finally:
                # Always signal completion so the driver doesn’t hang.
                out_q.put(None)

    results: Dict[str, Dict[str, float]] = {}
    LOG_EVERY = int(os.environ.get("LOG_EVERY", "10"))  # progress write interval (examples)
    N_TOTAL = len(data)
    source_root = Path(__file__).resolve().parent.parent
    implementation_sha256 = hashlib.sha256(
        b"".join(
            path.read_bytes()
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("core.py").resolve(),
                source_root / "handoff_mas.py",
                source_root / "agent.py",
                source_root / "answer_utils.py",
                source_root / "dataset_handler.py",
                source_root / "mas.py",
                source_root / "tooling.py",
            )
        )
    ).hexdigest()
    base_manifest = {
        "dataset": name_dataset,
        "data_count": N_TOTAL,
        "data_sha256": _stable_hash(data),
        "agent_specs": worker_init.agent_specs,
        "edges": worker_init.edges,
        "handoff_config": worker_init.handoff_config,
        "step_separator": worker_init.step_separator,
        "gen_model_id": worker_init.gen_model_id,
        "prm": _checkpoint_identity(worker_init.prm_dir),
        "prm_base_model_id": worker_init.prm_base_model_id,
        "orm": _checkpoint_identity(worker_init.orm_dir),
        "dtype": worker_init.dtype,
        "attn_implementation": worker_init.attn_impl,
        "seed": worker_init.seed,
        "implementation_sha256": implementation_sha256,
    }
    handoff_kinds = HANDOFF_KINDS
    handoff_bundle_needs_prm = any(spec.kind == "handoff_prm" for spec in conditions)
    handoff_workers = None

    for spec in conditions:
        condition_needs_prm = spec.kind in {
            "sbs_prm",
            "mcts_prm",
            "voter_prm",
            "handoff_prm",
        }
        worker_needs_prm = condition_needs_prm or (
            spec.kind in handoff_kinds and handoff_bundle_needs_prm
        )
        needs_orm = spec.kind in {"mcts_orm", "voter_orm"}
        needs_judge = spec.kind == "mcts_judge"
        spec_worker_init = WorkerInit(
            agent_specs=worker_init.agent_specs,
            edges=worker_init.edges,
            handoff_config=worker_init.handoff_config,
            step_separator=worker_init.step_separator,
            gen_model_id=worker_init.gen_model_id,
            prm_dir=worker_init.prm_dir if worker_needs_prm else None,
            prm_base_model_id=(worker_init.prm_base_model_id if worker_needs_prm else None),
            orm_dir=worker_init.orm_dir if needs_orm else None,
            judge_model_id=worker_init.judge_model_id if needs_judge else None,
            judge_load_in_4bit=worker_init.judge_load_in_4bit,
            prm_max_length=int(spec.params.get("prm_context_length", worker_init.prm_max_length)),
            attn_impl=worker_init.attn_impl,
            dtype=worker_init.dtype,
            seed=worker_init.seed,
        )

        # Prepare conditional path parts
        prm = (
            f"_{Path(worker_init.prm_dir).name}"
            if worker_init.prm_dir and condition_needs_prm
            else ""
        )
        orm = f"_{Path(worker_init.orm_dir).name}" if worker_init.orm_dir and needs_orm else ""

        # Prepare model ID and a short stable fingerprint so distinct settings
        # (e.g. view_mode=full vs local) do not collide in the skip log path.
        model = worker_init.gen_model_id.split("/")[-1]
        params = "_".join(f"{k}-{v}" for k, v in spec.params.items())[:20]
        condition_manifest = {
            **base_manifest,
            "condition": {
                "name": spec.name,
                "kind": spec.kind,
                "params": spec.params,
            },
        }
        manifest_hash = _stable_hash(condition_manifest)
        params_fingerprint = manifest_hash[:12]

        # Assemble final path
        results_path = (
            f"logs/{name_dataset}_{spec.name}{params}_{params_fingerprint}"
            f"{prm}{orm}_{model}_{worker_init.seed}.txt"
        )
        records_path = str(Path(results_path).with_suffix(".jsonl"))
        manifest_path = str(Path(results_path).with_suffix(".manifest.json"))
        summary_path = str(Path(results_path).with_suffix(".summary.json"))
        os.makedirs("logs", exist_ok=True)
        Path(manifest_path).write_text(
            json.dumps(
                {"manifest_hash": manifest_hash, "manifest": condition_manifest},
                indent=2,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )
        if _completed_cache_is_valid(
            results_path,
            summary_path,
            records_path,
            manifest_hash=manifest_hash,
            expected_count=N_TOTAL,
        ):
            results[spec.name] = json.loads(Path(summary_path).read_text())
            print(f"[skip] Loaded completed '{spec.name}' from {results_path}.")
            continue
        if Path(results_path).exists():
            print(f"[rerun] Resetting incomplete cache for '{spec.name}'.")

        # A rerun must not retain a stale COMPLETED marker or summary.
        with open(results_path, "w") as f:
            f.write(
                f"=== START {spec.name} | dataset={name_dataset} | N={N_TOTAL} "
                f"| manifest={manifest_hash} ===\n"
            )
        Path(records_path).write_text("")
        Path(summary_path).unlink(missing_ok=True)

        if spec.kind not in handoff_kinds and handoff_workers is not None:
            for worker in handoff_workers:
                ray.kill(worker, no_restart=True)
            handoff_workers = None
        if spec.kind in handoff_kinds and handoff_workers is not None:
            workers = handoff_workers
        else:
            workers = [
                EvalWorker.remote(spec_worker_init, worker_init.dtype, rank=i)
                for i in range(num_workers)
            ]
            if spec.kind in handoff_kinds:
                handoff_workers = workers
        in_q = Queue()
        out_q = Queue()
        consume_refs = [w.consume.remote(spec, in_q, out_q) for w in workers]

        base_seed = int(getattr(worker_init, "seed", 0) or 0)
        for idx, ex in enumerate(data):
            ex_seed = base_seed + idx
            in_q.put((idx, ex["question"], ex["answer"], ex_seed))
        for _ in range(len(workers)):
            in_q.put(None)

        correct = 0  # pass@1 on the best prediction
        pass_counts = [0] * MAX_PASS_K  # pass@k over candidate sets
        totals: List[int] = []
        finished = 0
        worker_crashed = False
        prm_calls_total = 0
        agent_runs_total = 0
        prefix_correct = {4: 0, 8: 0, 12: 0}
        handoff_examples = 0
        depths: List[int] = []
        unique_agents: List[int] = []
        unique_edges: List[int] = []
        revisit_rates: List[float] = []
        route_entropies: List[float] = []
        route_counts_by_speaker: Dict[str, Counter] = {}
        context_token_values: List[int] = []
        context_truncations = 0
        parse_failures_total = 0
        proposals_total = 0
        prm_evaluations_total = 0
        decoder_errors = 0
        latencies: List[float] = []
        judge_calls_total = 0
        judge_parse_failures_total = 0
        tool_usage_totals: Counter = Counter()
        tool_names_total: Counter = Counter()
        pbar = tqdm(total=len(data), desc=f"Evaluating: {spec.name}")
        while finished < len(workers):
            try:
                # Try to get an item with a timeout (e.g., 5 seconds)
                item = out_q.get(timeout=5)
            except queue.Empty:
                ready_refs, _ = ray.wait(consume_refs, timeout=0)

                if ready_refs:
                    # If a worker is "ready" (finished) but we didn't get a None in the queue,
                    for ref in ready_refs:
                        try:
                            ray.get(ref)  # This will raise the worker's exception if it crashed
                        except Exception as e:
                            print(f"\n[CRITICAL] A worker crashed! Error: {e}")
                            worker_crashed = True
                            finished += 1
                            consume_refs.remove(ref)
                continue
            if item is None:
                finished += 1
                continue
            idx, pred, cand_list, tok_total, prm_calls, agent_runs, metadata, gold = item

            # Normalize candidate list and ensure pred is first
            if not isinstance(cand_list, (list, tuple)):
                cand_list = [cand_list]
            ordered_cands: List[str] = [pred] + [c for c in cand_list if c != pred]
            if not ordered_cands:
                ordered_cands = [pred]
            ordered_cands = ordered_cands[:MAX_PASS_K]

            # pass@1 (accuracy on top-1)
            top_correct = is_correct(pred, gold)
            if top_correct:
                correct += 1

            if isinstance(metadata, dict) and "prefix_answers" in metadata:
                handoff_examples += 1
                for threshold in prefix_correct:
                    prefix_answer = metadata.get("prefix_answers", {}).get(str(threshold), "")
                    prefix_is_correct = is_correct(prefix_answer, gold)
                    prefix_correct[threshold] += int(prefix_is_correct)
                depths.append(int(metadata.get("depth", 0)))
                unique_agents.append(int(metadata.get("unique_agents", 0)))
                unique_edges.append(int(metadata.get("unique_directed_role_edges", 0)))
                revisit_rates.append(float(metadata.get("role_revisit_rate", 0.0)))
                route_entropies.append(float(metadata.get("route_entropy_bits", 0.0)))
                for turn in metadata.get("turns", []):
                    speaker = str(turn.get("speaker", ""))
                    recipient = str(turn.get("recipient", ""))
                    if speaker and recipient and recipient != "FINAL":
                        route_counts_by_speaker.setdefault(speaker, Counter())[recipient] += 1
                context_token_values.extend(
                    int(value) for value in metadata.get("prm_context_tokens", [])
                )
                context_truncations += int(metadata.get("prm_truncations", 0))
                parse_failures_total += int(metadata.get("parse_failures", 0))
                proposals_total += int(metadata.get("agent_proposals", 0))
                prm_evaluations_total += int(metadata.get("prm_evaluations", 0))
            if isinstance(metadata, dict) and "latency_s" in metadata:
                latencies.append(float(metadata["latency_s"]))
            if isinstance(metadata, dict) and metadata.get("judge_calls"):
                judge_calls_total += int(metadata.get("judge_calls", 0))
                judge_parse_failures_total += int(metadata.get("judge_parse_failures", 0))
            if isinstance(metadata, dict) and isinstance(metadata.get("tool_usage"), dict):
                tool_usage = metadata["tool_usage"]
                for key, value in tool_usage.items():
                    if key != "tool_names":
                        if key == "max_prompt_tokens":
                            tool_usage_totals[str(key)] = max(
                                tool_usage_totals[str(key)], int(value)
                            )
                        else:
                            tool_usage_totals[str(key)] += int(value)
                for tool_name, count in tool_usage.get("tool_names", {}).items():
                    tool_names_total[str(tool_name)] += int(count)
            if isinstance(metadata, dict) and metadata.get("error"):
                decoder_errors += 1

            record = {
                "index": int(idx),
                "condition": spec.kind,
                "manifest_hash": manifest_hash,
                "prediction": str(pred),
                "gold": str(gold),
                "correct": bool(top_correct),
                "candidates": [str(value) for value in ordered_cands],
                "metrics": metadata if isinstance(metadata, dict) else {},
            }
            with open(records_path, "a") as records_file:
                records_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            # pass@k: any correct among the first k candidates
            any_correct = False
            for ki in range(MAX_PASS_K):
                if ki < len(ordered_cands) and is_correct(ordered_cands[ki], gold):
                    any_correct = True
                if any_correct:
                    pass_counts[ki] += 1

            if tok_total is not None:
                totals.append(int(tok_total))
            prm_calls_total += int(prm_calls or 0)
            agent_runs_total += int(agent_runs or 0)
            pbar.update(1)

            # periodic progress logging
            seen = len(totals)
            if seen % LOG_EVERY == 0:
                acc_so_far = correct / max(1, seen)
                mean_so_far = float(statistics.mean(totals)) if totals else 0.0
                std_so_far = float(statistics.pstdev(totals)) if len(totals) > 1 else 0.0
                pass_str = ", ".join(
                    f"p@{k}={pass_counts[k-1]/max(1, seen):.4f}"
                    for k in REPORTED_PASS_K
                )
                with open(results_path, "a") as f:
                    tool_enabled = tool_usage_totals["tool_enabled_candidates"]
                    tool_used = tool_usage_totals["tool_used_candidates"]
                    model_responses = tool_usage_totals["model_responses"]
                    truncated_responses = tool_usage_totals["truncated_responses"]
                    f.write(
                        "[progress] "
                        f"{seen}/{N_TOTAL} "
                        f"acc={acc_so_far:.4f} "
                        f"{pass_str} "
                        f"tokens={mean_so_far:.1f}±{std_so_far:.1f} "
                        f"prm_calls_total={prm_calls_total} "
                        f"agent_runs_total={agent_runs_total} "
                        f"prm_calls_mean={prm_calls_total/max(1, seen):.2f} "
                        f"agent_runs_mean={agent_runs_total/max(1, seen):.2f} "
                        f"tool_use={tool_used/max(1, tool_enabled):.4f} "
                        f"trunc={truncated_responses/max(1, model_responses):.4f}\n"
                    )

        pbar.close()

        # Ensure outstanding tasks are resolved before tearing down this pool.
        if consume_refs:
            ray.get(consume_refs)
        if spec.kind not in handoff_kinds:
            for w in workers:
                ray.kill(w, no_restart=True)
        for queue_actor in (in_q, out_q):
            try:
                queue_actor.shutdown()
            except Exception:
                pass

        processed = len(totals)
        mean = float(statistics.mean(totals)) if totals else 0.0
        std = float(statistics.pstdev(totals)) if len(totals) > 1 else 0.0
        tool_enabled = int(tool_usage_totals["tool_enabled_candidates"])
        tool_used = int(tool_usage_totals["tool_used_candidates"])
        tool_calls = int(tool_usage_totals["tool_calls"])
        successful_tool_calls = int(tool_usage_totals["successful_tool_calls"])
        model_responses = int(tool_usage_totals["model_responses"])
        model_prompt_tokens = int(tool_usage_totals["model_prompt_tokens"])
        truncated_responses = int(tool_usage_totals["truncated_responses"])
        tool_summary = {
            "tool_enabled_candidates_total": tool_enabled,
            "tool_used_candidates_total": tool_used,
            "tool_use_rate": tool_used / tool_enabled if tool_enabled else 0.0,
            "tool_model_responses_total": model_responses,
            "model_prompt_tokens_total": model_prompt_tokens,
            "model_prompt_tokens_mean": (
                model_prompt_tokens / model_responses if model_responses else 0.0
            ),
            "max_prompt_tokens": int(tool_usage_totals["max_prompt_tokens"]),
            "truncated_responses_total": truncated_responses,
            "response_truncation_rate": (
                truncated_responses / model_responses if model_responses else 0.0
            ),
            "tool_calls_total": tool_calls,
            "successful_tool_calls_total": successful_tool_calls,
            "successful_tool_call_rate": (
                successful_tool_calls / tool_calls if tool_calls else 0.0
            ),
            "tool_errors_total": int(tool_usage_totals["tool_errors"]),
            "invalid_tool_calls_total": int(tool_usage_totals["invalid_tool_calls"]),
            "required_tool_misses_total": int(tool_usage_totals["required_tool_misses"]),
            "forced_tool_reprompts_total": int(tool_usage_totals["forced_tool_reprompts"]),
            "tool_names": dict(sorted(tool_names_total.items())),
        }

        if worker_crashed or processed != len(data) or decoder_errors:
            with open(results_path, "a") as f:
                f.write(
                    f"FAILED: processed {processed}/{len(data)} examples; "
                    f"worker_crashed={worker_crashed}\n"
                )
                f.write(f"Decoder errors: {decoder_errors}\n")
                f.write(f"Tokens so far: {mean:.1f} ± {std:.1f}\n")
                f.write(f"Total PRM calls so far: {prm_calls_total}\n")
                f.write(f"Total Agent runs so far: {agent_runs_total}\n")
            results[spec.name] = {
                "accuracy": float("nan"),
                "tok_mean": mean,
                "tok_std": std,
                "prm_calls_total": int(prm_calls_total),
                "prm_evaluations_total": int(prm_evaluations_total),
                "agent_runs_total": int(agent_runs_total),
                "prm_calls_mean": float("nan"),
                "agent_runs_mean": float("nan"),
                **tool_summary,
            }
            for k in REPORTED_PASS_K:
                results[spec.name][f"pass_at_{k}"] = float("nan")
            print(f"[result] {spec.name}: FAILED " f"({processed}/{len(data)} examples processed)")
            if spec.kind in handoff_kinds and handoff_workers is not None:
                for worker in handoff_workers:
                    ray.kill(worker, no_restart=True)
                handoff_workers = None
            continue

        n = len(data)
        acc = correct / n
        pass_at = [c / n for c in pass_counts]

        results[spec.name] = {
            "accuracy": acc,
            "tok_mean": mean,
            "tok_std": std,
            "prm_calls_total": int(prm_calls_total),
            "prm_evaluations_total": int(prm_evaluations_total),
            "agent_runs_total": int(agent_runs_total),
            "prm_calls_mean": float(prm_calls_total) / n,
            "prm_evaluations_mean": float(prm_evaluations_total) / n,
            "agent_runs_mean": float(agent_runs_total) / n,
            **tool_summary,
        }
        for k in REPORTED_PASS_K:
            results[spec.name][f"pass_at_{k}"] = pass_at[k - 1]

        if handoff_examples:
            sorted_context = sorted(context_token_values)
            p95_idx = max(0, math.ceil(0.95 * len(sorted_context)) - 1) if sorted_context else 0
            routed_edge_total = sum(
                sum(recipient_counts.values())
                for recipient_counts in route_counts_by_speaker.values()
            )
            conditional_route_entropy = 0.0
            for recipient_counts in route_counts_by_speaker.values():
                speaker_total = sum(recipient_counts.values())
                speaker_entropy = 0.0
                for count in recipient_counts.values():
                    probability = count / speaker_total
                    speaker_entropy -= probability * math.log2(probability)
                conditional_route_entropy += (
                    speaker_total / max(1, routed_edge_total)
                ) * speaker_entropy
            handoff_summary = {
                f"hit_at_{threshold}": prefix_correct[threshold] / handoff_examples
                for threshold in prefix_correct
            }
            handoff_summary.update(
                {
                    "realized_depth_mean": float(statistics.mean(depths)),
                    "realized_depth_median": float(statistics.median(depths)),
                    "unique_agents_mean": float(statistics.mean(unique_agents)),
                    "unique_directed_role_edges_mean": float(statistics.mean(unique_edges)),
                    "role_revisit_rate_mean": float(statistics.mean(revisit_rates)),
                    "route_entropy_bits": conditional_route_entropy,
                    "within_trajectory_edge_entropy_bits_mean": float(
                        statistics.mean(route_entropies)
                    ),
                    "prm_context_token_median": (
                        float(statistics.median(sorted_context)) if sorted_context else 0.0
                    ),
                    "prm_context_token_p95": (
                        float(sorted_context[p95_idx]) if sorted_context else 0.0
                    ),
                    "prm_truncation_rate": (
                        context_truncations / len(context_token_values)
                        if context_token_values
                        else 0.0
                    ),
                    "parse_failure_rate": (
                        parse_failures_total / proposals_total if proposals_total else 0.0
                    ),
                    "agent_proposals_total": float(proposals_total),
                    "agent_proposals_mean": proposals_total / handoff_examples,
                }
            )
            results[spec.name].update(handoff_summary)

        if latencies:
            results[spec.name]["latency_mean_s"] = float(statistics.mean(latencies))
        if judge_calls_total:
            results[spec.name].update(
                {
                    "judge_calls_total": judge_calls_total,
                    "judge_calls_mean": judge_calls_total / n,
                    "judge_parse_failure_rate": (judge_parse_failures_total / judge_calls_total),
                }
            )

        Path(summary_path).write_text(
            json.dumps(results[spec.name], indent=2, sort_keys=True) + "\n"
        )

        # Append final summary and mark as completed
        with open(results_path, "a") as f:
            f.write(f"Accuracy (pass@1): {acc:.4f}\n")
            for k in REPORTED_PASS_K[1:]:
                f.write(f"Pass@{k}: {pass_at[k-1]:.4f}\n")
            f.write(f"Tokens: {mean:.1f} ± {std:.1f} ({len(totals)} examples)\n")
            f.write(f"Total PRM calls: {prm_calls_total}\n")
            f.write(f"Total Agent runs: {agent_runs_total}\n")
            f.write(f"Mean PRM calls per example: {results[spec.name]['prm_calls_mean']:.2f}\n")
            f.write(f"Mean Agent runs per example: {results[spec.name]['agent_runs_mean']:.2f}\n")
            f.write(
                "Model prompt tokens (mean/max): "
                f"{tool_summary['model_prompt_tokens_mean']:.1f}/"
                f"{tool_summary['max_prompt_tokens']}\n"
            )
            f.write(
                "Output truncations: "
                f"{tool_summary['truncated_responses_total']}/"
                f"{tool_summary['tool_model_responses_total']} "
                f"({tool_summary['response_truncation_rate']:.4f})\n"
            )
            if tool_enabled:
                f.write(
                    "Required-tool use: "
                    f"{tool_used}/{tool_enabled} "
                    f"({results[spec.name]['tool_use_rate']:.4f})\n"
                )
                f.write(
                    "Successful tool calls: "
                    f"{successful_tool_calls}/{tool_calls} "
                    f"({results[spec.name]['successful_tool_call_rate']:.4f})\n"
                )
                f.write(f"Tool calls by name: {dict(sorted(tool_names_total.items()))}\n")
                f.write(
                    "Tool errors / invalid calls / required misses / forced reprompts: "
                    f"{tool_summary['tool_errors_total']} / "
                    f"{tool_summary['invalid_tool_calls_total']} / "
                    f"{tool_summary['required_tool_misses_total']} / "
                    f"{tool_summary['forced_tool_reprompts_total']}\n"
                )
            if handoff_examples:
                f.write(f"Hit@1 at 4 messages: {results[spec.name]['hit_at_4']:.4f}\n")
                f.write(f"Hit@1 at 8 messages: {results[spec.name]['hit_at_8']:.4f}\n")
                f.write(f"Hit@1 at 12 messages: {results[spec.name]['hit_at_12']:.4f}\n")
                f.write(
                    "Realized depth (mean/median): "
                    f"{results[spec.name]['realized_depth_mean']:.2f}/"
                    f"{results[spec.name]['realized_depth_median']:.2f}\n"
                )
                f.write(
                    "Unique agents / directed edges (mean): "
                    f"{results[spec.name]['unique_agents_mean']:.2f}/"
                    f"{results[spec.name]['unique_directed_role_edges_mean']:.2f}\n"
                )
                f.write(
                    f"Role revisit rate (mean): {results[spec.name]['role_revisit_rate_mean']:.4f}\n"
                )
                f.write(
                    "Route entropy H(recipient|speaker), bits: "
                    f"{results[spec.name]['route_entropy_bits']:.4f}\n"
                )
                f.write(
                    "PRM context tokens (median/p95): "
                    f"{results[spec.name]['prm_context_token_median']:.1f}/"
                    f"{results[spec.name]['prm_context_token_p95']:.1f}\n"
                )
                f.write(f"PRM truncation rate: {results[spec.name]['prm_truncation_rate']:.4f}\n")
                f.write(
                    f"Action parse-failure rate: {results[spec.name]['parse_failure_rate']:.4f}\n"
                )
                f.write(
                    "Batched policy calls / proposals per example: "
                    f"{results[spec.name]['agent_runs_mean']:.2f}/"
                    f"{results[spec.name]['agent_proposals_mean']:.2f}\n"
                )
                f.write(
                    "Batched PRM calls / successor evaluations per example: "
                    f"{results[spec.name]['prm_calls_mean']:.2f}/"
                    f"{results[spec.name]['prm_evaluations_mean']:.2f}\n"
                )
            if latencies:
                f.write(
                    f"Selector latency mean (s): " f"{results[spec.name]['latency_mean_s']:.3f}\n"
                )
            if judge_calls_total:
                f.write(
                    f"Judge calls: {judge_calls_total} | parse-failure rate: "
                    f"{results[spec.name]['judge_parse_failure_rate']:.4f}\n"
                )
            f.write("COMPLETED\n")

        pass_str = " ".join(f"p@{k}={pass_at[k-1]:.4f}" for k in REPORTED_PASS_K)
        print(
            f"[result] {spec.name}: acc={acc:.4f} ({pass_str})  "
            f"tokens={mean:.1f}±{std:.1f} ({len(totals)} examples)"
        )

    if handoff_workers is not None:
        for worker in handoff_workers:
            ray.kill(worker, no_restart=True)

    return results
