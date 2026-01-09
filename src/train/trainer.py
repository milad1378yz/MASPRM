import os
import re
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Reduce CUDA fragmentation on long runs
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128",
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    BitsAndBytesConfig,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import load_file

from trl import PRMConfig, PRMTrainer
from data_collector import DataCollatorForTokenRegression, PairwiseDataCollatorForPPM

# Integer encoding for continuous labels
INT_LABEL_SCALE = 10_000  # e.g., y=-0.1234 -> -1234
INT_LABEL_SHIFT = 1_000_000  # offset so no real label ever equals -100

os.environ["HF_HOME"] = "./cache"


def arg_parser():
    ap = argparse.ArgumentParser("Train PRM (regression), ORM (last-step PRM) or PPM (pairwise).")
    # Mode
    ap.add_argument(
        "--mode",
        choices=["prm", "ppm", "orm"],
        default="prm",
        help="prm: PRM/PQM regression; ppm: pairwise Process Preference Model; orm: PRM but only the last step.",
    )
    # Data
    ap.add_argument(
        "--data_dir",
        type=Path,
        default=Path("results/prm_samples_gsm8k_res_train_Qwen_Qwen2.5-1.jsonl"),
        help="Path to JSON/JSONL file",
    )
    # Model & training
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--output-dir", type=Path, default=Path("checkpoints/Qwen2.5-1.5B-RM"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument(
        "--hub-logs",
        action="store_true",
        default=True,
        help="Report to tensorboard if set",
    )
    ap.add_argument("--seed", type=int, default=42)
    # Sequence packing
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-prompt-length", type=int, default=768)
    ap.add_argument("--step-separator", type=str, default="</step>")
    ap.add_argument(
        "--ensure-step-token",
        action="store_true",
        default=True,
        help="If using a sentinel like '<extra_0>', add it to tokenizer.",
    )
    # Loss (PRM/ORM only)
    ap.add_argument("--loss", choices=["mse", "huber", "bce"], default="huber")
    ap.add_argument("--huber-beta", type=float, default=0.1)
    # LoRA
    ap.add_argument("--lora-r", type=int, default=256)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA/QLoRA and quantization; full finetune with standard AdamW.",
    )

    args = ap.parse_args()
    return args


def _is_main_process():
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except ValueError:
        return True


def rank_zero_print(*args, **kwargs):
    if _is_main_process():
        print(*args, **kwargs)


# PRM (regression) trainer
class PRMRegressionTrainer(PRMTrainer):
    """
    Continuous PRM in [-1, 1] using logit-space loss:
      z_target = atanh(y)  (with clamp)
      loss = MSE/Huber(z, z_target) on separator positions only
    """

    def __init__(self, *args, loss_type: str = "mse", huber_beta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_type = loss_type

        if loss_type == "huber":
            self._loss = nn.SmoothL1Loss(beta=huber_beta)
        elif loss_type == "mse":
            self._loss = nn.MSELoss()
        elif loss_type == "bce":
            self._loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        # Resolve the separator to *exact* token ids via encode, not convert_tokens_to_ids
        sep = getattr(self.args, "step_separator", "</step>")
        tok = self.processing_class  # this is the tokenizer
        sep_ids = tok.encode(sep, add_special_tokens=False)

        # Require the separator to be a SINGLE token for clean supervision
        if len(sep_ids) != 1:
            raise AssertionError(
                f"Step separator '{sep}' is not a single token (ids={sep_ids}). "
                f"Add it as a special token or use --ensure-step-token."
            )
        self._sep_id = sep_ids[0]
        self._checked_once = False

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")  # (B, T) in [-1,1], -100 mask elsewhere
        ids = inputs["input_ids"]  # (B, T)

        # One-time hard checks for alignment
        if not self._checked_once:
            # per-example: #labels == #separators
            B = labels.size(0)
            for b in range(B):
                num_labels = int((labels[b] != -100).sum().item())
                num_seps = int((ids[b] == self._sep_id).sum().item())
                if num_labels != num_seps:
                    raise AssertionError(
                        f"Example {b}: labels ({num_labels}) != separators ({num_seps}). "
                        "Ensure you only place labels at the step separator."
                    )
            # labeled positions must be exactly the separator token
            wrong = ((labels != -100) & (ids != self._sep_id)).any().item()
            if wrong:
                raise AssertionError(
                    "PRM labels are not placed exactly on the step separator token."
                )
            self._checked_once = True

        out = model(**inputs)
        z = out.logits.squeeze(-1).to(torch.float32)  # logits (B, T)
        mask = labels != -100

        # Guard: if truncation produced an empty mask for this batch, return a 0-loss
        if not torch.any(mask):
            raise AssertionError("All examples in the batch have no labeled positions.")

        # Labels are already float targets in [-1,1] from the collator
        y = labels.to(torch.float32)

        # Stable logit-space targets (avoid tanh saturation / vanishing grads)
        eps = 0.05  # keep away from ±1 so atanh is finite and well-conditioned
        y = y.clamp(-1 + eps, 1 - eps)
        z_target = torch.atanh(y)
        loss = self._loss(z[mask], z_target[mask])
        return (loss, out) if return_outputs else loss

    @staticmethod
    def tokenize_row(
        features,
        tokenizer,
        step_separator,
        max_length=None,
        max_prompt_length=None,
        max_completion_length=None,
        train_on_last_step_only=False,
        is_eval=False,
    ):
        # 1) Get input_ids + an *integer* mask in row["labels"] from stock PRM tokenization.
        row = PRMTrainer.tokenize_row(
            features,
            tokenizer,
            step_separator,
            max_length,
            max_prompt_length,
            max_completion_length,
            train_on_last_step_only,
            is_eval,
        )

        # 2) Overwrite row["labels"] with float targets aligned to the mask (no new columns).
        int_mask = torch.tensor(row["labels"], dtype=torch.long)  # int mask from PRM
        mask = int_mask != -100

        if "labels" not in features or not isinstance(features["labels"], (list, tuple)):
            raise AssertionError(
                "Expect features['labels'] to be a list/sequence of floats in [-1,1]."
            )

        cont_vals = torch.tensor(features["labels"], dtype=torch.float32)
        if int(mask.sum().item()) != len(cont_vals):
            raise AssertionError(
                f"Mismatch: found {int(mask.sum().item())} labeled positions but {len(cont_vals)} provided labels."
            )

        # Quantize: y \in [-1,1] → int code = round(y * SCALE) + SHIFT
        encoded = torch.round(cont_vals * INT_LABEL_SCALE).to(torch.long) + INT_LABEL_SHIFT
        int_mask[mask] = encoded
        row["labels"] = int_mask.tolist()  # stays int64 for Arrow/TRL
        return row


# ORM trainer (PRM but only last step)
class ORMRegressionTrainer(PRMRegressionTrainer):
    """
    ORM = PRM trained only on the last step.
    Uses PRM-style data (multiple labels), but ignores all except the last one.
    """

    # Override the per-batch loss checks to accept exactly one labeled token,
    # and ensure it sits on the *last* step separator when present.
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")  # (B, T) in [-1,1], -100 mask elsewhere
        ids = inputs["input_ids"]  # (B, T)

        if not getattr(self, "_orm_checked_once", False):
            B = labels.size(0)
            for b in range(B):
                mask = labels[b] != -100
                nlab = int(mask.sum().item())
                if nlab == 0:
                    # Truncated example with no step label; allowed (will zero-loss)
                    continue
                if nlab != 1:
                    raise AssertionError(
                        f"ORM expects exactly one labeled position per example (got {nlab}). "
                        "Set train_on_last_step_only or check tokenization."
                    )
                # Must be on the separator and must be the *last* separator
                lab_idx = int(mask.nonzero(as_tuple=True)[0][-1].item())
                if int(ids[b, lab_idx].item()) != self._sep_id:
                    raise AssertionError("ORM label is not placed on the step separator token.")
                sep_pos = (ids[b] == self._sep_id).nonzero(as_tuple=True)[0]
                if sep_pos.numel() > 0 and lab_idx != int(sep_pos[-1].item()):
                    raise AssertionError("ORM label must be on the LAST step separator token.")
            self._orm_checked_once = True

        out = model(**inputs)

        z = out.logits.squeeze(-1).to(torch.float32)  # (B, T)
        mask = labels != -100

        # Empty after truncation -> neutral loss (do not crash)
        if not torch.any(mask):
            raise AssertionError("All examples in the batch have no labeled positions.")

        # Labels are already float targets in [-1,1] from the collator
        y = labels.to(torch.float32)

        # Branch on loss type
        if getattr(self, "_loss_type", None) == "bce":
            # y is in {-1, 1}; map to {0,1} for BCE
            targets = (y > 0).to(torch.float32)
            loss = self._loss(z[mask], targets[mask])
        else:
            # original regression-in-logit-space behavior
            eps = 0.05
            y = y.clamp(-1 + eps, 1 - eps)
            z_target = torch.atanh(y)
            loss = self._loss(z[mask], z_target[mask])
        return (loss, out) if return_outputs else loss

    @staticmethod
    def tokenize_row(
        features,
        tokenizer,
        step_separator,
        max_length=None,
        max_prompt_length=None,
        max_completion_length=None,
        train_on_last_step_only=False,  # ignored; always True for ORM
        is_eval=False,
    ):
        # Ask the base PRM to build a mask only for the *last* step.
        row = PRMTrainer.tokenize_row(
            features,
            tokenizer,
            step_separator,
            max_length,
            max_prompt_length,
            max_completion_length,
            True,  # force: only last step is labeled
            is_eval,
        )

        # Overwrite row["labels"] with a single float target at the last-step position.
        int_mask = torch.tensor(row["labels"], dtype=torch.long)
        mask = int_mask != -100  # either one position or 0 if truncated

        if mask.any():
            if "labels" not in features or len(features["labels"]) == 0:
                raise AssertionError("ORM expects PRM-style 'labels'; got empty.")
            last_val = float(features["labels"][-1])  # y ∈ [-1,1]
            encoded = int(round(last_val * INT_LABEL_SCALE)) + INT_LABEL_SHIFT
            int_mask[mask] = encoded

        row["labels"] = int_mask.tolist()  # keep as ints
        return row


def compute_regression_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.tanh(np.asarray(preds).squeeze(-1))
    labels = np.asarray(labels)
    mask = labels != -100
    yhat = preds[mask].astype(np.float32)
    # Labels already decoded to [-1,1] by the collator
    y = labels[mask].astype(np.float32)

    err = yhat - y
    mse = float(np.mean(err**2)) if y.size else 0.0
    rmse = float(np.sqrt(mse)) if y.size else 0.0
    mae = float(np.mean(np.abs(err))) if y.size else 0.0
    bias = float(np.mean(err)) if y.size else 0.0
    if y.size > 1:
        y_mean = float(np.mean(y))
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        if ss_tot <= 1e-12:
            perfect = ss_res <= 1e-12
            r2 = explained_var = 1.0 if perfect else 0.0
        else:
            r2 = 1.0 - ss_res / ss_tot
            explained_var = 1.0 - float(np.var(err)) / float(np.var(y))
    else:
        r2 = explained_var = 0.0
    return {
        "rmse": rmse,
        "pearson": float(np.corrcoef(yhat, y)[0, 1]) if y.size > 1 else 0.0,
        "r2": r2,
        "explained_variance": explained_var,
        "mae": mae,
        "mse": mse,
        "bias": bias,
    }


# PPM (pairwise preference) utilities
def _join_steps(prompt: str, steps: List[str], sep: str) -> str:
    """Build a single text where each step ends with the separator token."""
    text = prompt.rstrip() + "\n"
    for s in steps:
        s = s.strip()
        if s:
            text += s + " " + sep + " "
    return text


def _expand_ppm_pairs(base: Dataset, sep: str) -> Dataset:
    """
    From a base dataset with fields:
      { "prompt": str, "chosen_actions": [str...], "rejected_actions": [str...] }
    produce a dataset of pairwise comparisons:
      { "pos_text": str, "neg_text": str }
    where each pair differs at the first differing step.
    """
    pairs: List[Dict[str, str]] = []
    for row in base:
        prompt = row["prompt"]
        chosen = [s.strip() for s in row["chosen_actions"]]
        rejected = [s.strip() for s in row["rejected_actions"]]
        n = min(len(chosen), len(rejected))
        if n == 0:
            continue
        diff_idx = -1
        for i in range(n - 1, -1, -1):
            if chosen[i] != rejected[i]:
                diff_idx = i
                break
        if diff_idx < 0:
            continue
        prefix = chosen[:diff_idx]
        pairs.append(
            {
                "pos_text": _join_steps(prompt, prefix + [chosen[diff_idx]], sep),
                "neg_text": _join_steps(prompt, prefix + [rejected[diff_idx]], sep),
            }
        )
    return Dataset.from_list(pairs)


def _ppm_tokenize_fn(
    examples: Dict[str, List[str]], tokenizer: AutoTokenizer, max_length: int
) -> Dict[str, Any]:
    pos = tokenizer(
        examples["pos_text"],
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    neg = tokenizer(
        examples["neg_text"],
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=True,
    )
    return {
        "pos_input_ids": pos["input_ids"],
        "pos_attention_mask": pos["attention_mask"],
        "neg_input_ids": neg["input_ids"],
        "neg_attention_mask": neg["attention_mask"],
    }


def _last_sep_logits(
    logits: torch.Tensor, input_ids: torch.Tensor, sep_id: int, pad_id: int
) -> torch.Tensor:
    """
    logits: (B, T, 1) or (B, T)  ; input_ids: (B, T)
    Returns (B,) tensor with logit taken at last occurrence of sep_id in each row;
    fallback to last non-pad token if no separator is present after truncation.
    """
    if logits.dim() == 3:
        logits = logits.squeeze(-1)
    B, T = input_ids.shape
    idx = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
    is_sep = input_ids == sep_id
    last_idx = torch.where(is_sep, idx, torch.full_like(idx, -1)).max(dim=1).values  # (B,)
    # Fallback: last non-pad token
    last_tok = (input_ids != pad_id).int().sum(dim=1) - 1
    last_idx = torch.where(last_idx >= 0, last_idx, last_tok.clamp(min=0))
    return logits.gather(1, last_idx.view(-1, 1)).squeeze(1)


def compute_ppm_metrics(eval_pred):
    arr = np.asarray(eval_pred.predictions)
    # Backward compatibility if you keep the old path somewhere
    if arr.ndim == 1:
        d = arr
        pos = neg = p_found = n_found = None
    else:
        d, pos, neg, p_found, n_found = (
            arr[:, 0],
            arr[:, 1],
            arr[:, 2],
            arr[:, 3],
            arr[:, 4],
        )

    out = {}
    if d.size:
        out["pair_acc"] = float((d > 0).mean())
        out["avg_margin"] = float(d.mean())
        out["median_margin"] = float(np.median(d))
        out["std_margin"] = float(d.std())
        out["avg_bt_prob"] = float((1.0 / (1.0 + np.exp(-d))).mean())  # mean \sigma(diff)

    if pos is not None:
        out["pos_mean"] = float(pos.mean())
        out["neg_mean"] = float(neg.mean())

    if p_found is not None:
        out["fallback_rate_pos"] = float((1.0 - p_found).mean())
        out["fallback_rate_neg"] = float((1.0 - n_found).mean())
        out["fallback_rate_any"] = float(((p_found < 0.5) | (n_found < 0.5)).mean())

    out["num_pairs"] = int(d.size)
    return out


def _last_sep_logits_with_flag(
    logits: torch.Tensor, input_ids: torch.Tensor, sep_id: int, pad_id: int
):
    if logits.dim() == 3:
        logits = logits.squeeze(-1)
    B, T = input_ids.shape
    idx = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
    is_sep = input_ids == sep_id
    last_idx = torch.where(is_sep, idx, torch.full_like(idx, -1)).max(dim=1).values  # (B,)
    found = last_idx >= 0  # whether we saw at least one separator
    # Fallback to last non-pad
    last_tok = (input_ids != pad_id).int().sum(dim=1) - 1
    last_idx = torch.where(found, last_idx, last_tok.clamp(min=0))
    scores = logits.gather(1, last_idx.view(-1, 1)).squeeze(1)
    return scores, found


class PPMTrainer(Trainer):
    """
    Pairwise Bradley-Terry ranking loss for step-level preference training:

        L = - E[ log \sigma( r(x, y_pos) - r(x, y_neg) ) ]

    Here, r(x, y) is the single score taken at the last step separator token.
    """

    def __init__(self, *args, tokenizer: AutoTokenizer, step_separator: str = "</step>", **kwargs):
        super().__init__(*args, processing_class=tokenizer, **kwargs)
        sep_ids = tokenizer.encode(step_separator, add_special_tokens=False)
        if len(sep_ids) != 1:
            raise AssertionError(
                f"Step separator '{step_separator}' must be a single token (ids={sep_ids}). "
                f"Consider --ensure-step-token to add it as an additional special token."
            )
        self._sep_id = sep_ids[0]
        self._pad_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        pos_ids = inputs["pos_input_ids"]
        pos_mask = inputs["pos_attention_mask"]
        neg_ids = inputs["neg_input_ids"]
        neg_mask = inputs["neg_attention_mask"]

        out_pos = model(input_ids=pos_ids, attention_mask=pos_mask)
        out_neg = model(input_ids=neg_ids, attention_mask=neg_mask)
        pos_score = _last_sep_logits(
            out_pos.logits.to(torch.float32), pos_ids, self._sep_id, self._pad_id
        )  # (B,)
        neg_score = _last_sep_logits(
            out_neg.logits.to(torch.float32), neg_ids, self._sep_id, self._pad_id
        )  # (B,)

        diff = pos_score - neg_score
        loss = -F.logsigmoid(diff).mean()  # Bradley–Terry / InstructGPT-style pairwise loss

        if return_outputs:
            # return "logits" as the difference so metrics can use it
            return loss, type("Out", (), {"logits": diff.detach()})
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            # pick up the dummy labels injected by the collator
            labels = inputs.get("labels", None)

            pos_ids = inputs["pos_input_ids"]
            pos_mask = inputs["pos_attention_mask"]
            neg_ids = inputs["neg_input_ids"]
            neg_mask = inputs["neg_attention_mask"]

            out_pos = model(input_ids=pos_ids, attention_mask=pos_mask)
            out_neg = model(input_ids=neg_ids, attention_mask=neg_mask)

            pos_score, pos_found = _last_sep_logits_with_flag(
                out_pos.logits.to(torch.float32), pos_ids, self._sep_id, self._pad_id
            )
            neg_score, neg_found = _last_sep_logits_with_flag(
                out_neg.logits.to(torch.float32), neg_ids, self._sep_id, self._pad_id
            )

            diff = pos_score - neg_score
            loss = -F.logsigmoid(diff).mean()

            preds = torch.stack(
                [diff, pos_score, neg_score, pos_found.float(), neg_found.float()],
                dim=1,
            )

        # IMPORTANT: return labels, not None
        return (loss, preds, labels)


# Misc helpers
def find_last_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.exists():
        return None
    ckpts = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def main():
    is_main = _is_main_process()

    args = arg_parser()
    use_lora = not args.no_lora
    rank_zero_print(f"LoRA enabled: {use_lora}")

    # Load dataset
    full_ds = load_dataset("json", data_files={"train": str(args.data_dir)})["train"]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        trust_remote_code=True,
        # local_files_only=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optionally ensure the step separator is a known/special token
    if args.ensure_step_token:
        looks_like_special = bool(re.match(r"^<[^>\s]+>$", args.step_separator))
        if (
            looks_like_special
            and tokenizer.convert_tokens_to_ids(args.step_separator) == tokenizer.unk_token_id
        ):
            tokenizer.add_special_tokens({"additional_special_tokens": [args.step_separator]})
            rank_zero_print(f"Added special token to tokenizer: {args.step_separator}")
    rank_zero_print(
        "Separator ids:",
        tokenizer.encode(args.step_separator, add_special_tokens=False),
    )

    # QLoRA config (V100-friendly)
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        if use_lora
        else None
    )

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            # "embed_tokens",
        ],
        bias="all",
        # bias="none",
        modules_to_save=[
            "score",
            "norm",
            "input_layernorm",
            "post_attention_layernorm",
            "classifier",
            "embed_tokens",
        ],
    )

    # Robust dtype/attention selection for V100-class GPUs (no BF16 support).
    if torch.cuda.is_available():
        cc_major = torch.cuda.get_device_capability(0)[0]
        ampere_or_newer = cc_major >= 8
        from torch.nn.attention import sdpa_kernel, SDPBackend

        if ampere_or_newer:
            # Ampere+ → allow efficient kernels
            sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
            attn_impl = "sdpa"
        else:
            # Volta/Turing → math or eager only
            sdpa_kernel([SDPBackend.MATH])
            attn_impl = "sdpa"
    else:
        cc_major = 0
        ampere_or_newer = False
        attn_impl = "sdpa"  # Default for CPU

    # LoRA path: 4-bit  FP16 compute as before. Full FT path: never BF16 unless Ampere.
    if use_lora:
        chosen_dtype = torch.float16
    else:
        chosen_dtype = torch.bfloat16 if ampere_or_newer else torch.float16

    device_map = "auto" if use_lora else None

    print(f"Using torch_dtype={chosen_dtype}, attn_impl={attn_impl}, device_map={device_map}")

    # Model with single‑logit token head
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_id,
        num_labels=1,
        device_map=device_map,
        quantization_config=bnb_config,
        torch_dtype=chosen_dtype,
        attn_implementation=attn_impl,
        # local_files_only=True,
    )
    model.config.use_cache = False

    if use_lora:
        # Gradient checkpointing is great with QLoRA; keep it on.
        if torch.cuda.is_available():
            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        # Older kernels/drivers are prone to invalid-argument errors with gc + FP32.
        if torch.cuda.is_available():
            model.gradient_checkpointing_disable()

    # Resize embeddings if we added special tokens
    if args.ensure_step_token and tokenizer.vocab_size != model.get_input_embeddings().weight.size(
        0
    ):
        model.resize_token_embeddings(len(tokenizer))
        rank_zero_print("Resized token embeddings to match tokenizer.")

    # Optimizer choice depends on LoRA/quantization
    optim_name = "adamw_bnb_8bit" if use_lora else "adamw_torch"

    # Common training args template
    common_cfg = dict(
        output_dir=str(args.output_dir),
        do_train=True,
        do_eval=True,
        auto_find_batch_size=True,
        # per_device_train_batch_size=32,
        # per_device_eval_batch_size=64,
        gradient_accumulation_steps=args.grad_accum,
        bf16=(chosen_dtype == torch.bfloat16),
        fp16=(chosen_dtype == torch.float16),
        optim=optim_name,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        step_separator=args.step_separator,
        dataloader_num_workers=4,
        dataloader_persistent_workers=False,
        report_to=("tensorboard" if args.hub_logs else None),
        average_tokens_across_devices=True,
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    if args.mode == "ppm":
        # Keep rightmost tokens so last step separator survives truncation
        tokenizer.truncation_side = "left"

        expanded = _expand_ppm_pairs(full_ds, args.step_separator)
        splits = expanded.train_test_split(test_size=0.04, seed=args.seed)
        train_dataset, eval_dataset = splits["train"], splits["test"]

        base = Path(args.output_dir) / "map_cache"
        base.mkdir(parents=True, exist_ok=True)

        train_cache = (
            base
            / f"ppm_tok_{tokenizer.name_or_path.replace('/','-')}_L{args.max_length}_train.arrow"
        )
        eval_cache = (
            base
            / f"ppm_tok_{tokenizer.name_or_path.replace('/','-')}_L{args.max_length}_eval.arrow"
        )

        train_dataset = train_dataset.map(
            _ppm_tokenize_fn,
            batched=True,
            remove_columns=train_dataset.column_names,
            fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
            cache_file_name=str(train_cache),
            load_from_cache_file=True,
        )
        eval_dataset = eval_dataset.map(
            _ppm_tokenize_fn,
            batched=True,
            remove_columns=eval_dataset.column_names,
            fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
            cache_file_name=str(eval_cache),
            load_from_cache_file=True,
        )

        if use_lora:
            model = get_peft_model(model, lora_config)

        training_args = PRMConfig(
            **common_cfg,
            metric_for_best_model="loss",
            greater_is_better=False,
            group_by_length=False,
            remove_unused_columns=False,  # already needed for PPM collator
            label_names=["labels"],
        )

        base_collator = PairwiseDataCollatorForPPM(
            tokenizer, padding=True, max_length=args.max_length
        )

        # Wrap it to add a dummy labels vector so Trainer collects label_ids.
        class _PPMCollatorWithDummyLabel:
            def __init__(self, base):
                self.base = base

            def __call__(self, features):
                batch = self.base(features)
                B = batch["pos_input_ids"].size(0)
                batch["labels"] = torch.zeros(B, dtype=torch.long)
                return batch

        data_collator = _PPMCollatorWithDummyLabel(base_collator)

        trainer = PPMTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            step_separator=args.step_separator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_ppm_metrics,
        )

    elif args.mode == "orm":
        # Keep rightmost tokens to preserve the *last* step separator under truncation
        tokenizer.truncation_side = "left"

        splits = full_ds.train_test_split(test_size=0.04, seed=args.seed)
        train_dataset, eval_dataset = splits["train"], splits["test"]

        training_args = PRMConfig(
            **common_cfg,
            metric_for_best_model="pearson",
            greater_is_better=True,
            group_by_length=True,
            eval_accumulation_steps=16,
            # Ensure the collator can see 'targets'
            remove_unused_columns=False,
            # For clarity; ORM trainer forces last-step supervision anyway.
            train_on_last_step_only=True,
        )

        data_collator = DataCollatorForTokenRegression(tokenizer, label_pad_token_id=-100.0)
        peft_cfg = lora_config if use_lora else None
        trainer = ORMRegressionTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_cfg,
            compute_metrics=compute_regression_metrics,
            loss_type=args.loss,
            huber_beta=args.huber_beta,
            data_collator=data_collator,
        )

    else:
        # PRM / PQM regression path
        tokenizer.truncation_side = "left"
        splits = full_ds.train_test_split(test_size=0.04, seed=42)
        train_dataset, eval_dataset = splits["train"], splits["test"]

        training_args = PRMConfig(
            **common_cfg,
            metric_for_best_model="pearson",
            greater_is_better=True,
            group_by_length=True,
            eval_accumulation_steps=16,
            # Ensure the collator can see 'targets'
            remove_unused_columns=False,
        )

        data_collator = DataCollatorForTokenRegression(tokenizer, label_pad_token_id=-100.0)
        peft_cfg = lora_config if use_lora else None
        trainer = PRMRegressionTrainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_cfg,
            compute_metrics=compute_regression_metrics,
            loss_type=args.loss,
            huber_beta=args.huber_beta,
            data_collator=data_collator,
        )

    # Count params
    total, trainable = 0, 0
    for _, p in trainer.model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    rank_zero_print(f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")

    # Train / resume (weights only, using safetensors; no torch.load)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Sync before scanning filesystem
    if dist.is_initialized():
        dist.barrier()
    last_ckpt_path = find_last_checkpoint(args.output_dir)
    resume_ckpt = str(last_ckpt_path) if last_ckpt_path is not None else None
    rank_zero_print(f"Last checkpoint directory (weights only): {resume_ckpt}")

    # If we found a checkpoint, try to load safetensor weights manually
    if resume_ckpt is not None:
        ckpt_files = [
            os.path.join(resume_ckpt, "adapter_model.safetensors"),  # LoRA / QLoRA
            os.path.join(resume_ckpt, "model.safetensors"),  # full-model safetensors
        ]
        loaded = False
        for ckpt_file in ckpt_files:
            if os.path.exists(ckpt_file):
                state = load_file(ckpt_file)
                missing, unexpected = trainer.model.load_state_dict(state, strict=False)
                rank_zero_print(f"Loaded weights from: {ckpt_file}")
                if missing:
                    rank_zero_print(f"Missing keys (first 5): {missing[:5]}")
                if unexpected:
                    rank_zero_print(f"Unexpected keys (first 5): {unexpected[:5]}")
                loaded = True
                break

        if not loaded:
            rank_zero_print(
                "Checkpoint dir exists but no *.safetensors weights found; "
                "training will start from the base model."
            )

    # Small help against fragmentation before heavy allocations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if dist.is_initialized():
        dist.barrier()

    # IMPORTANT: do NOT pass resume_from_checkpoint here → avoids torch.load / CVE guard
    trainer.train()

    # Evaluate
    final = trainer.evaluate()
    rank_zero_print("Final eval metrics:", final)

    # Save
    if is_main:
        trainer.save_model(str(args.output_dir))
        rank_zero_print(f"Saved model/peft adapters to: {args.output_dir.resolve()}")

    # Clean teardown
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
