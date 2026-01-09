from datasets import load_dataset
import re
from fractions import Fraction
import random


def load_hard_dataset(name, split, n, seed):
    if name == "competition_math":
        ds = load_dataset("qwedsacf/competition_math")
        shuffled = ds["train"].shuffle(seed=seed)
        # Define explicit train/test split
        test_size = 2000
        train_data = shuffled.select(range(test_size, len(shuffled)))
        test_data = shuffled.select(range(test_size))

        if split == "train":
            data = train_data.select(range(n)) if n else train_data
        elif split == "test":
            data = test_data.select(range(n)) if n else test_data
        else:
            raise ValueError(f"Unknown split: {split}")

        gold_fn = gold_competition_math
        q_fn = lambda ex: ex["problem"]

    elif name == "aqua_rat":
        ds = load_dataset("aqua_rat")
        data = ds[split].shuffle(seed=seed).select(range(n)) if n else ds[split]
        gold_fn = gold_aqua_rat
        q_fn = lambda ex: ex["question"]

    elif name == "gsm8k":
        ds = load_dataset("gsm8k", "main")
        data = ds[split].shuffle(seed=seed).select(range(n)) if n else ds[split]
        gold_fn = lambda ex: _to_number(
            re.findall(r"####\s*([-\$]?\s*\d[\d,]*(?:\.\d+)?(?:\s+\w+)?)", ex["answer"])[0]
        )
        q_fn = lambda ex: ex["question"]

    elif name == "svamp":
        ds = load_dataset("svamp")
        data = ds[split].shuffle(seed=seed).select(range(n)) if n else ds[split]
        gold_fn = gold_svamp
        q_fn = lambda ex: ex["Body"] + " " + ex["Question"]

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

    elif name == "gpqa":
        # Official GPQA Main split from HuggingFace:
        #   - HF id: "Idavidrein/gpqa"
        #   - config: "gpqa_main"
        #   - split: "train"
        # Note: the dataset is gated – you must accept its terms on HF before
        # this will download successfully.
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main")

        shuffled = ds["train"].shuffle(seed=seed)
        # Define explicit train/test split
        test_size = 100
        train_data = shuffled.select(range(test_size, len(shuffled)))
        test_data = shuffled.select(range(test_size))

        if split == "train":
            data = train_data.select(range(n)) if n else train_data
        elif split == "test":
            data = test_data.select(range(n)) if n else test_data
        else:
            raise ValueError(f"Unknown split: {split}")

        gold_fn = gold_gpqa
        q_fn = format_gpqa_question

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
            "Unsupported dataset. Choose from: competition_math, aqua_rat, svamp, gsm8k, mmlu, gpqa, logiqa"
        )

    return data, q_fn, gold_fn


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


def gold_aqua_rat(example):
    # multiple choice; correct option index (e.g., 'A', 'B', ...) and options text.
    key = example["correct"]
    # Some entries store the final numeric inside options like "A) 42"
    opts = example["options"]
    # Map key to option string
    label_to_idx = {c: i for i, c in enumerate("ABCDE")}
    if key in label_to_idx and 0 <= label_to_idx[key] < len(opts):
        return _to_number(re.sub(r"^[A-E]\)\s*", "", opts[label_to_idx[key]]))
    return None


def gold_svamp(example):
    # short numeric answer in 'Answer'
    return _to_number(str(example["Answer"]))


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


def _to_number(s):
    if s is None:
        return None
    s = s.strip()
    s = s.replace(",", "")
    s = re.sub(r"^\$", "", s)
    s = re.sub(r"\s+(dollars?|tickets?|units?|boxes?|people|students?)$", "", s, flags=re.I)
    if re.fullmatch(r"-?\d+/\d+", s):
        return float(Fraction(s))
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except:
            return None
    try:
        return float(s)
    except:
        return None


def _gpqa_perm(ex):
    """
    Deterministically shuffle the 4 answer options so that:
      - The model doesn't see 'correct is always A'.
      - The same GPQA example always gets the same option order.
    We derive a seed from Record ID (or fall back to Question text).
    """
    rid = ex.get("Record ID") or ex.get("Record_ID") or ex.get("Question") or ""
    rid_str = str(rid)

    # Simple deterministic seed: sum of character codes
    seed = sum(ord(c) for c in rid_str)
    rng = random.Random(seed)

    idxs = [0, 1, 2, 3]
    rng.shuffle(idxs)  # in-place
    return idxs


def format_gpqa_question(ex):
    """
    Turn a GPQA example into a multiple-choice prompt with A-D options.

    Fields used (from Idavidrein/gpqa, gpqa_main):
      - Question
      - Correct Answer
      - Incorrect Answer 1/2/3
      - Record ID (for permutation)
    """
    q = (ex.get("Question") or "").strip()

    # Original order: index 0 is the correct answer.
    opts = [
        ex.get("Correct Answer"),
        ex.get("Incorrect Answer 1"),
        ex.get("Incorrect Answer 2"),
        ex.get("Incorrect Answer 3"),
    ]

    # Fallback in case any field is missing (defensive)
    opts = ["" if o is None else str(o) for o in opts]

    perm = _gpqa_perm(ex)  # permutation of [0,1,2,3]
    labels = "ABCD"

    lines = [q]
    for new_i, old_i in enumerate(perm):
        lines.append(f"{labels[new_i]}. {opts[old_i]}")

    # Instruct the model to answer with a single letter, like for MMLU.
    return "\n".join(lines)


def gold_gpqa(ex):
    """
    Return the correct choice as a single uppercase letter A-D, consistent
    with format_gpqa_question's permutation.
    """
    perm = _gpqa_perm(ex)
    # In the original order, index 0 is the correct answer ('Correct Answer').
    correct_pos = perm.index(0)  # where did index 0 end up after shuffling?
    return "ABCD"[correct_pos]
