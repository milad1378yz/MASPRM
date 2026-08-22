from datasets import load_dataset
import re
from fractions import Fraction

from answer_utils import _to_number


_SOFT_NEWLINE_CODEPOINTS = dict.fromkeys((11, 12, 13, 0x85, 0x2028, 0x2029))
_SPLIT_SEED = 42


def _remove_soft_newlines(value):
    return str(value).translate(_SOFT_NEWLINE_CODEPOINTS)


def load_hard_dataset(name, split, n, seed):
    if name == "competition_math":
        ds = load_dataset("qwedsacf/competition_math")
        # The paper uses one fixed shuffle for this derived partition. Run
        # seeds affect model sampling, not which examples belong to each split.
        shuffled = ds["train"].shuffle(seed=_SPLIT_SEED)
        test_size = 2000
        train_data = shuffled.select(range(test_size, len(shuffled)))
        test_data = shuffled.select(range(test_size))

        if split == "train":
            data = train_data
        elif split == "test":
            data = test_data
        else:
            raise ValueError(f"Unknown split: {split}")

        if n:
            data = data.select(range(min(int(n), len(data))))

        gold_fn = gold_competition_math
        q_fn = lambda ex: ex["problem"]

    elif name == "gsm8k":
        ds = load_dataset("gsm8k", "main")
        data = ds[split].shuffle(seed=seed).select(range(n)) if n else ds[split]
        gold_fn = lambda ex: gold_from_gsm8k(ex["answer"])
        q_fn = lambda ex: ex["question"]

    elif name == "mgsm":
        ds = load_dataset("juletxara/mgsm", "en")
        if split not in ds:
            raise ValueError(f"Unknown split for mgsm: {split}")
        data = ds[split].shuffle(seed=seed).select(range(n)) if n else ds[split]
        gold_fn = lambda ex: _to_number(str(ex["answer_number"]))
        q_fn = lambda ex: ex["question"]

    elif name == "mmlu":

        ds = load_dataset("cais/mmlu", "all")

        # Map user-provided split to what's available in cais/mmlu
        split_alias = {
            "train": "test",
            "test": "validation",
        }
        real_split = split_alias.get(split, split)

        data_raw = ds[real_split]
        data = data_raw.shuffle(seed=seed).select(range(n)) if n else data_raw

        gold_fn = gold_mmlu
        q_fn = format_mmlu_question

    elif name == "logiqa":
        # Supports both `lucasmccabe/logiqa` (correct_option) and the original
        # script-based `logiqa` / EleutherAI/logiqa (label).
        ds = load_dataset("lucasmccabe/logiqa")

        # Map common split aliases → HF splits
        split_alias = {
            "train": "train",
            "test": "test",
        }
        real_split = split_alias.get(split, split)
        if real_split not in ds:
            raise ValueError(f"Unknown split for logiqa: {split}")

        data_raw = ds[real_split]
        data = data_raw.shuffle(seed=seed).select(range(n)) if n else data_raw

        gold_fn = gold_logiqa
        q_fn = format_logiqa_question

    else:
        raise ValueError(
            "Unsupported dataset. Choose from: competition_math, gsm8k, mgsm, "
            "mmlu, logiqa"
        )

    return data, lambda ex: _remove_soft_newlines(q_fn(ex)), gold_fn


def format_logiqa_question(ex):
    """
    Turn a LogiQA example into a multiple-choice prompt with A-D options.

    Supports both:
      - lucasmccabe/logiqa: {context, query, options, correct_option}
      - EleutherAI/logiqa: {context, question, options, label}
    """
    ctx = (ex.get("context") or "").strip()
    q = (ex.get("query") or ex.get("question") or "").strip()
    opts = ex.get("options") or []

    if not isinstance(opts, list):
        opts = list(opts)

    labels = "ABCD"
    lines = []
    if ctx:
        lines.append(f"Context: {ctx}")
    if q:
        lines.append(f"Question: {q}")

    for i, opt in enumerate(opts[:4]):
        lines.append(f"{labels[i]}. {opt}")

    return "\n".join(lines)


def gold_logiqa(ex):
    """
    Return the correct choice for LogiQA as a single uppercase letter A-D.

    Handles:
      - lucasmccabe/logiqa: correct_option = 0..3  (int)
      - EleutherAI/logiqa:  label = "0".."3" or "A".."D"
    """
    idx = ex.get("correct_option", None)
    if isinstance(idx, str) and idx.isdigit():
        idx = int(idx)

    if isinstance(idx, int):
        if 0 <= idx < 4:
            return "ABCD"[idx]

    # Fallback: original script version uses `label`
    lab = ex.get("label")
    if isinstance(lab, str):
        s = lab.strip()
        if s.isdigit() and 0 <= int(s) < 4:
            return "ABCD"[int(s)]
        if s.upper() in ["A", "B", "C", "D"]:
            return s.upper()

    return None


def gold_from_gsm8k(answer_field: str):
    m = re.search(r"####\s*([^\n]+)", answer_field)
    return _to_number(m.group(1)) if m else None


#  gold extractors for each dataset
def gold_competition_math(example):
    # solution usually ends with \boxed{<ans>} possibly with spaces or signs
    sol = example["solution"]
    m = re.findall(r"\\boxed\{([^}]+)\}", sol)
    if m:
        # last \boxed is the final answer
        x = m[-1]
        # strip LaTeX cruft like \frac{a}{b}
        if re.match(r"^\s*\\frac\{[-+]?\d+\}\{[-+]?\d+\}\s*$", x):
            num, den = re.findall(r"-?\d+", x)
            return float(Fraction(int(num), int(den)))
        # remove latex spacing
        x = re.sub(r"\\[a-zA-Z]+", "", x)
        x = x.replace("{", "").replace("}", "").strip()
        return _to_number(x)
    return None


def format_mmlu_question(ex):
    """Format a single MMLU item into a prompt with A-D options."""
    q = (ex.get("question") or ex.get("prompt") or ex.get("stem") or "").strip()

    # Options can appear as 'choices' (list) or as separate A/B/C/D fields.
    opts = ex.get("choices") or ex.get("options")
    if not opts:
        maybe = [ex.get(k) for k in ("A", "B", "C", "D")]
        opts = [o for o in maybe if o is not None]

    # Be robust to dict-wrapped choices (rare)
    if isinstance(opts, dict) and "text" in opts:
        opts = opts["text"]

    if not isinstance(opts, list):
        opts = list(opts) if opts is not None else []

    labels = "ABCD"
    lines = [q] + [f"{labels[i]}. {opt}" for i, opt in enumerate(opts[:4])]
    return "\n".join(lines) + "\nAnswer with a single letter (A, B, C, or D) only."


def gold_mmlu(ex):
    """
    Return the correct choice as a single uppercase letter A-D.
    Supports:
      - answer: 'A'|'B'|'C'|'D'
      - answer: int index (0..3)
      - answer text that exactly matches one of the options
    """
    ans = ex.get("answer") or ex.get("target") or ex.get("label") or ex.get("correct")
    # 1) numeric index
    if isinstance(ans, int):
        return "ABCD"[ans]

    # 2) already a letter
    if isinstance(ans, str):
        s = ans.strip()
        m = re.fullmatch(r"[A-Da-d]", s)
        if m:
            return s.upper()

    # 3) try to find which option matches the answer text
    opts = ex.get("choices") or ex.get("options")
    if not opts:
        maybe = [ex.get(k) for k in ("A", "B", "C", "D")]
        opts = [o for o in maybe if o is not None]
    if isinstance(opts, dict) and "text" in opts:
        opts = opts["text"]
    if isinstance(ans, str) and isinstance(opts, list):
        tgt = _norm(ans)
        for i, o in enumerate(opts[:4]):
            if _norm(o) == tgt:
                return "ABCD"[i]

    # As a last resort, try first character if it looks like 'A) ...'
    if isinstance(ans, str) and len(ans) >= 1 and ans[0].upper() in "ABCD":
        return ans[0].upper()

    return None


def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()

