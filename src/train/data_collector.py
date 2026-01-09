from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any
import torch
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from transformers.tokenization_utils_base import PaddingStrategy


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded


@dataclass
class DataCollatorForTokenRegression:
    """
    Collator for PRM/ORM regression.

    - If a 'targets' column is present in the dataset, it is used as the source of continuous labels.
    - Otherwise falls back to 'labels' or 'label'.
    - Returns batch['labels'] as float32 with -100.0 on unlabeled positions.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: float = -100.0
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict]):
        # 1) Choose the supervision source in order of preference.
        if "targets" in features[0]:
            source_name = "targets"  # preferred: float targets
        elif "labels" in features[0]:
            source_name = "labels"  # may be float labels OR integer-encoded floats OR a mask
        elif "label" in features[0]:
            source_name = "label"
        else:
            source_name = None

        labels = [f[source_name] for f in features] if source_name else None

        # 2) Strip label-like fields so pad() only sees model inputs.
        model_inputs = [
            {k: v for k, v in f.items() if k not in ("targets", "labels", "label")}
            for f in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            model_inputs,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        if labels is None:
            return batch

        # 3) Prepare to normalize/ decode into float32 labels.
        seq_len = batch["input_ids"].shape[1]
        pad_val = float(self.label_pad_token_id)
        padding_side = self.tokenizer.padding_side

        # Optional decode support for integer-encoded continuous labels.
        # Keep in sync with the constants used during encoding in your trainer.
        INT_LABEL_SCALE = 10_000
        INT_LABEL_SHIFT = 1_000_000

        def _looks_encoded_value(v: int) -> bool:
            # Heuristic: any large non -100 int indicates SCALE/SHIFT encoding.
            return isinstance(v, int) and v != -100 and v >= (INT_LABEL_SHIFT // 10)

        # If the source is not 'targets', check if any sequence looks encoded.
        need_decode = source_name != "targets" and any(
            any(_looks_encoded_value(v) for v in seq) for seq in labels
        )

        def _to_float_sequence(seq):
            out = []
            for v in seq:
                if v == -100 or v == -100.0:
                    out.append(pad_val)
                elif need_decode and isinstance(v, int) and _looks_encoded_value(v):
                    out.append((v - INT_LABEL_SHIFT) / float(INT_LABEL_SCALE))
                else:
                    out.append(float(v))
            return out

        float_labels = [_to_float_sequence(seq) for seq in labels]

        # 4) Pad labels to the input length, respecting tokenizer padding side.
        if padding_side == "right":
            padded = [row + [pad_val] * (seq_len - len(row)) for row in float_labels]
        else:
            padded = [[pad_val] * (seq_len - len(row)) + row for row in float_labels]

        # 5) Return as float32 under the canonical key 'labels'.
        batch["labels"] = torch.tensor(padded, dtype=torch.float32)
        return batch


class PairwiseDataCollatorForPPM:
    """Pads pos and neg batches separately, returns tensors."""

    def __init__(
        self, tokenizer: AutoTokenizer, padding=True, max_length=None, return_tensors="pt"
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.return_tensors = return_tensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pos_feats = [
            {"input_ids": f["pos_input_ids"], "attention_mask": f["pos_attention_mask"]}
            for f in features
        ]
        neg_feats = [
            {"input_ids": f["neg_input_ids"], "attention_mask": f["neg_attention_mask"]}
            for f in features
        ]

        pos_batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            pos_feats,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )
        neg_batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            neg_feats,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )

        return {
            "pos_input_ids": pos_batch["input_ids"],
            "pos_attention_mask": pos_batch["attention_mask"],
            "neg_input_ids": neg_batch["input_ids"],
            "neg_attention_mask": neg_batch["attention_mask"],
        }
