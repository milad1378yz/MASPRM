from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import os
import queue
import statistics
from pathlib import Path

import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import ray
from ray.util.queue import Queue
from openai import OpenAI

from answer_utils import is_correct
from agent import Agent
from mas import MAS
from core import (
    _seed_everything,
    TokenStats,
    sbs_decode2,
    make_scored_voter,
    load_prm_scorer,
    ensure_separator_token,
    align_model_to_tokenizer,
    load_generation_model,
    MCTSInfer,
    _majority_by_numbers_equal,
)


# =========================================================
#                  Ray worker + evaluator
# =========================================================


@dataclass
class WorkerInit:
    # MAS graph
    agent_specs: List[Dict[str, Any]]
    edges: List[List[int]]
    step_separator: str = "</step>"

    # Models
    gen_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    prm_dir: Optional[str] = None
    prm_base_model_id: Optional[str] = None
    orm_dir: Optional[str] = None

    # OpenAI config
    use_openai: bool = False
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None

    # dtype options: "float16" | "bfloat16" | "float32"
    dtype: str = "float16"
    seed: int = 0


@dataclass
class ConditionSpec:
    name: str
    kind: str  # 'sbs_none' | 'sbs_prm' | 'sbs_logprob' | 'mcts_prm' | 'mcts_logprob' | 'mcts_orm' | 'voter_plain' | 'voter_prm' | 'voter_logprob' | 'voter_orm'
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


def evaluate_conditions_ray(
    data: List[Dict[str, str]],
    worker_init: WorkerInit,
    conditions: List[ConditionSpec],
    name_dataset: str,
) -> Dict[str, Dict[str, float]]:
    """
    Create a Ray worker pool ONCE (models initialized in each worker),
    then evaluate multiple conditions in series while reusing that pool.

    Returns: {name: {"accuracy": float, "tok_mean": float, "tok_std": float}}
    """
    # We will compute pass@k for these k values (k is over *independent runs* of the method)
    MAX_PASS_K = 10

    # --- Ray init (idempotent)
    if not ray.is_initialized():
        # Best-effort working_dir/PYTHONPATH for cluster workers
        src_dir = str(Path(__file__).resolve().parent.parent)  # .../src
        existing = os.environ.get("PYTHONPATH", "")

        env_vars_to_pass = {
            "PYTHONPATH": f"{existing}:{src_dir}" if existing else src_dir,
            # Pass the cache location to the workers
            "HF_HOME": os.environ.get("HF_HOME", "/tmp/hf"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", ""),
            # Pass the offline mode setting
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "0"),
        }

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
            self.dtype = _dtype_from_str(dtype)

            # 1) Load PRM scorer (optional)
            self.prm_score_fn = None
            self.prm_tok = None
            self.prm_model = None
            if init.prm_dir:
                self.prm_score_fn, self.prm_tok, self.prm_model = load_prm_scorer(
                    init.prm_dir,
                    base_model_id=init.prm_base_model_id,
                    torch_dtype=self.dtype,
                )

                # modify the PRM tokenizer/model with the separator
                added = ensure_separator_token(self.prm_tok, self.step_sep)
                print(f"[Worker {self.rank}] Added {added} special tokens for step separator.")
                if added > 0 and self.prm_model is not None:
                    align_model_to_tokenizer(self.prm_model, self.prm_tok)
            # 2) Load generation model; reuse PRM tokenizer if present to keep vocab identical
            self.use_openai = init.use_openai
            self.openai_client = None
            self.openai_model_name = (
                init.gen_model_id
            )  # Use gen_model_id as the OpenAI model name (e.g. gpt-4)

            if self.use_openai:
                self.policy_model = None
                self.policy_tok = AutoTokenizer.from_pretrained(
                    init.gen_model_id if "/" in init.gen_model_id else "Qwen/Qwen2.5-1.5B-Instruct",
                    use_fast=True,
                    trust_remote_code=True,
                )
                self.openai_client = OpenAI(
                    api_key=init.openai_api_key, base_url=init.openai_base_url
                )

            else:
                # Original logic
                self.policy_model, self.policy_tok = load_generation_model(
                    init.gen_model_id,
                    tokenizer=None,
                    torch_dtype=self.dtype,
                )
            # # 3) Ensure step token present in tokenizer & resize both heads if added
            # added = ensure_separator_token(self.prm_tok, self.step_sep)
            # print(f"[Worker {self.rank}] Added {added} special tokens for step separator.")
            # # Only align if we actually have a local model
            # if added > 0 and self.policy_model is not None:
            #     align_model_to_tokenizer(self.policy_model, self.policy_tok)
            # if added > 0 and self.prm_model is not None:
            #     align_model_to_tokenizer(self.prm_model, self.prm_tok)

            # 4) ORM scorer (optional) — same API as PRM loader
            self.orm_score_fn = None
            self.orm_tok = None
            if init.orm_dir:
                self.orm_score_fn, self.orm_tok, _ = load_prm_scorer(
                    init.orm_dir,
                    base_model_id=None,  # read from adapter config
                    torch_dtype=self.dtype,
                )

        # --- build MAS graph for each decode
        def _build_mas(self) -> MAS:
            agents = []
            for spec in self.agent_specs:
                agents.append(
                    Agent(
                        self.policy_model,
                        self.policy_tok,
                        system_prompt=spec.get("system_prompt", ""),
                        max_new_tokens=int(spec.get("max_new_tokens", 512)),
                        # OpenAI args
                        use_openai=self.use_openai,
                        openai_client=self.openai_client,
                        openai_model=self.openai_model_name,
                    )
                )
            return MAS(self.edges, agents)

        # --- decoders by spec.kind
        def _decode_by_spec(
            self, q: str, spec: ConditionSpec
        ) -> tuple[str, List[str], int, int, int]:
            kind = spec.kind
            p = spec.params or {}

            # Defaults
            sbs_kwargs = p.get(
                "gen_kwargs", {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 1024}
            )
            mcts_kwargs = p.get(
                "mcts_kwargs", {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 1024}
            )
            B1 = int(p.get("B1", 1))
            B2 = int(p.get("B2", 5))
            n_sims = int(p.get("n_simulations", 40))
            max_children = int(p.get("max_children", 3))
            c_uct = float(p.get("c_uct", 2.0))
            k = int(p.get("k", 5))

            pass_k = max(1, int(p.get("pass_k", 5)))

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
            mas0 = self._build_mas()

            #  SBS variants
            if kind == "sbs_none":
                finals: List[str] = []
                pred, usage = sbs_decode2(
                    mas0,
                    q,
                    score_type="none",
                    B1=B1,
                    B2=B2,
                    gen_kwargs=sbs_kwargs,
                    step_separator=self.step_sep,
                    final_answers_out=finals,
                )
                candidates = finals[:pass_k] if finals else [pred]
                return pred, candidates, usage.generated, usage.prm_calls, usage.agent_runs

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
                )
                candidates = finals[:pass_k] if finals else [pred]
                return pred, candidates, usage.generated, usage.prm_calls, usage.agent_runs

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
                    logprob_agg=p.get("logprob_agg", "sum"),
                    final_answers_out=finals,
                )
                candidates = finals[:pass_k] if finals else [pred]
                return pred, candidates, usage.generated, usage.prm_calls, usage.agent_runs

            #  MCTS variants
            if kind in ("mcts_prm", "mcts_logprob", "mcts_orm"):
                score_type = {"mcts_prm": "prm", "mcts_logprob": "logprob", "mcts_orm": "orm"}[kind]
                score_fn = (
                    self.prm_score_fn
                    if score_type == "prm"
                    else (self.orm_score_fn if score_type == "orm" else None)
                )
                prm_tok = self.prm_tok if score_type == "prm" else None
                orm_tok = self.orm_tok if score_type == "orm" else None

                leaf_type = p.get("leaf_score_type", None)
                leaf_fn = (
                    self.orm_score_fn
                    if (leaf_type == "orm")
                    else (self.prm_score_fn if (leaf_type == "prm") else score_fn)
                )

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
                    uct_type=p.get("uct_type", "uct"),
                    leaf_score_type=p.get("leaf_score_type", None),
                    logprob_agg=p.get("logprob_agg", "sum"),
                    leaf_score_fn=leaf_fn,
                )
                pred, usage = infer.decode(n_simulations=n_sims)
                try:
                    candidates = infer.get_topk_answers(k=pass_k)
                except Exception:
                    candidates = []
                if not candidates:
                    candidates = [pred]
                return pred, candidates, usage.generated, usage.prm_calls, usage.agent_runs

            #  Majority voters
            if kind == "voter_plain":
                preds: List[str] = []
                total = TokenStats()
                p0, u0, _ = base_policy_with_trace(mas0, q)
                preds.append(p0)
                total.add(u0)
                for _ in range(max(0, k - 1)):
                    mas_i = self._build_mas()
                    pi, ui, _ = base_policy_with_trace(mas_i, q)
                    preds.append(pi)
                    total.add(ui)
                pred = _majority_by_numbers_equal(preds)
                candidates = preds[:pass_k] if pass_k > 0 else [pred]
                return pred, candidates, total.generated, total.prm_calls, total.agent_runs

            if kind in ("voter_prm", "voter_logprob", "voter_orm"):
                # Build make_scored_voter *inside* the actor (no cross-process capture)
                score_mode = {"voter_prm": "prm", "voter_logprob": "logprob", "voter_orm": "orm"}[
                    kind
                ]

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
                    self._build_mas,
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
                    logprob_agg=p.get("logprob_agg", "sum"),
                    return_all_answers=True,
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
                return pred, candidates, usage.generated, usage.prm_calls, usage.agent_runs

            # Fallback: do a single greedy
            finals: List[str] = []
            pred, usage = sbs_decode2(
                mas0,
                q,
                score_type="none",
                B1=1,
                B2=1,
                gen_kwargs=sbs_kwargs,
                step_separator=self.step_sep,
                final_answers_out=finals,  # NEW
            )
            candidates = finals[:pass_k] if finals else [pred]
            return pred, candidates, usage.generated, usage.prm_calls, usage.agent_runs

        # --- queue consumer
        def consume(self, spec: ConditionSpec, in_q: "Queue", out_q: "Queue"):
            try:
                while True:
                    item = in_q.get()
                    if item is None:
                        break
                    idx, q, gold, ex_seed = item
                    _seed_everything(int(ex_seed))
                    try:
                        res = self._decode_by_spec(q, spec)
                        # normalize length (be paranoid)
                        if isinstance(res, tuple):
                            if len(res) == 5:
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
                    out_q.put((idx, pred, cand_list, tok_total, prm_calls, agent_runs, gold))
            finally:
                # Always signal completion so the driver doesn’t hang.
                out_q.put(None)

    # --- Create worker pool ONCE (reused across conditions)
    workers = [
        EvalWorker.remote(worker_init, worker_init.dtype, rank=i) for i in range(num_workers)
    ]

    # --- Evaluate all conditions, reusing the pool
    results: Dict[str, Dict[str, float]] = {}
    LOG_EVERY = int(os.environ.get("LOG_EVERY", "10"))  # progress write interval (examples)
    N_TOTAL = len(data)
    for spec in conditions:
        # Prepare conditional path parts
        prm = (
            f"_{Path(worker_init.prm_dir).name}"
            if worker_init.prm_dir and spec.kind in {"sbs_prm", "mcts_prm", "voter_prm"}
            else ""
        )
        orm = (
            f"_{Path(worker_init.orm_dir).name}"
            if worker_init.orm_dir and spec.kind in {"mcts_orm", "voter_orm"}
            else ""
        )

        # Prepare model ID and truncated params
        model = worker_init.gen_model_id.split("/")[-1]
        params = "_".join(f"{k}-{v}" for k, v in spec.params.items())[:20]

        # Assemble final path
        results_path = (
            f"logs/{name_dataset}_{spec.name}{params}{prm}{orm}_{model}_{worker_init.seed}.txt"
        )
        os.makedirs("logs", exist_ok=True)
        # If file already contains a COMPLETED marker, skip this condition
        if Path(results_path).exists():
            try:
                txt = Path(results_path).read_text()
            except Exception:
                txt = ""
            if "COMPLETED" in txt:
                print(f"[skip] Found completed log for '{spec.name}' at {results_path}. Skipping.")
                continue
        # Write a start header (append if rerunning)
        with open(results_path, "a") as f:
            f.write(f"=== START {spec.name} | dataset={name_dataset} | N={N_TOTAL} ===\n")

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
        prm_calls_total = 0
        agent_runs_total = 0
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
                            finished += 1
                            consume_refs.remove(ref)
                continue
            if item is None:
                finished += 1
                continue
            idx, pred, cand_list, tok_total, prm_calls, agent_runs, gold = item

            # Normalize candidate list and ensure pred is first
            if not isinstance(cand_list, (list, tuple)):
                cand_list = [cand_list]
            ordered_cands: List[str] = [pred] + [c for c in cand_list if c != pred]
            if not ordered_cands:
                ordered_cands = [pred]
            ordered_cands = ordered_cands[:MAX_PASS_K]

            # pass@1 (accuracy on top-1)
            if is_correct(pred, gold):
                correct += 1

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
                    f"p@{i+1}={pass_counts[i]/max(1, seen):.4f}" for i in range(MAX_PASS_K)
                )
                with open(results_path, "a") as f:
                    f.write(
                        "[progress] "
                        f"{seen}/{N_TOTAL} "
                        f"acc={acc_so_far:.4f} "
                        f"{pass_str} "
                        f"tokens={mean_so_far:.1f}±{std_so_far:.1f} "
                        f"prm_calls_total={prm_calls_total} "
                        f"agent_runs_total={agent_runs_total} "
                        f"prm_calls_mean={prm_calls_total/max(1, seen):.2f} "
                        f"agent_runs_mean={agent_runs_total/max(1, seen):.2f}\n"
                    )

        pbar.close()

        # Ensure tasks done (actors persist for next spec)
        ray.get(consume_refs)

        n = max(1, len(data))
        acc = correct / n
        pass_at = [c / n for c in pass_counts]
        mean = float(statistics.mean(totals)) if totals else 0.0
        std = float(statistics.pstdev(totals)) if len(totals) > 1 else 0.0

        results[spec.name] = {
            "accuracy": acc,
            "tok_mean": mean,
            "tok_std": std,
            "prm_calls_total": int(prm_calls_total),
            "agent_runs_total": int(agent_runs_total),
            "prm_calls_mean": float(prm_calls_total) / n,
            "agent_runs_mean": float(agent_runs_total) / n,
        }
        for i, v in enumerate(pass_at, start=1):
            results[spec.name][f"pass_at_{i}"] = v

        # Append final summary and mark as completed
        with open(results_path, "a") as f:
            f.write(f"Accuracy (pass@1): {acc:.4f}\n")
            for i in range(2, MAX_PASS_K + 1):
                f.write(f"Pass@{i}: {pass_at[i-1]:.4f}\n")
            f.write(f"Tokens: {mean:.1f} ± {std:.1f} ({len(totals)} examples)\n")
            f.write(f"Total PRM calls: {prm_calls_total}\n")
            f.write(f"Total Agent runs: {agent_runs_total}\n")
            f.write(f"Mean PRM calls per example: {results[spec.name]['prm_calls_mean']:.2f}\n")
            f.write(f"Mean Agent runs per example: {results[spec.name]['agent_runs_mean']:.2f}\n")
            f.write("COMPLETED\n")

        pass_str = " ".join(f"p@{i+1}={pass_at[i]:.4f}" for i in range(MAX_PASS_K))
        print(
            f"[result] {spec.name}: acc={acc:.4f} ({pass_str})  "
            f"tokens={mean:.1f}±{std:.1f} ({len(totals)} examples)"
        )

    # Optionally keep workers alive (faster for follow-ups). If you want to free VRAM at end:
    # for w in workers: ray.kill(w, no_restart=True)

    return results
