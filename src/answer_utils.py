import re
from fractions import Fraction
from typing import Optional


def normalize_answer(ans: str):
    ans = re.sub(r"\\boxed\{(.*?)\}", r"\1", ans)
    ans = ans.replace("$", "").replace("\\(", "").replace("\\)", "")
    ans = ans.strip()
    m = re.search(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?", ans.replace(" ", ""))
    if not m:
        return ans
    token = m.group(0)
    if "/" in token:
        try:
            return Fraction(token)
        except Exception:
            pass
    try:
        return Fraction(token)
    except Exception:
        return token


FINAL_ANSWER_RE = re.compile(
    r"""
    (?:\*\*|\b)?              # optional markdown bold start or word boundary
    final\s*answer            # "final answer"
    (?:\s*[:\-\u2013]?\s*|\s+is\s+)?  # optional colon/dash or "is"
    \(?                       # optional opening parenthesis
    ([ABCD])                  # the actual answer letter
    \)?                       # optional closing parenthesis
    (?:\*\*|\b)?              # optional markdown bold end or word boundary
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_final_letter(text: str) -> Optional[str]:
    """
    Extract a single A/B/C/D from free-form model output.
    Priority:
      1) 'final answer: X' (robust to **bold**, parentheses, or 'is X')
      2) output is exactly one letter
      3) first standalone A-D token anywhere (lenient fallback)
    """
    if not isinstance(text, str):
        return None
    # Try explicit "Final Answer" patterns first
    m = FINAL_ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    # If the whole text is just one letter
    m2 = re.fullmatch(r"\s*([ABCD])\s*", text.strip(), flags=re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    # Lenient fallback: first standalone A-D token
    m3 = re.search(r"\b([ABCD])\b", text.upper())
    return m3.group(1) if m3 else None


def answers_equal(pred, truth) -> bool:
    # Special-case MMLU: if the gold is exactly a letter A-D, parse a letter from pred.
    if isinstance(truth, str) and re.fullmatch(r"[ABCDabcd]", truth.strip()):
        gold = truth.strip().upper()
        pred_letter = extract_final_letter(str(pred))
        return (pred_letter is not None) and (pred_letter == gold)
    # Default math / text normalization:
    p = normalize_answer(str(pred))
    t = normalize_answer(str(truth))
    if isinstance(p, Fraction) and isinstance(t, Fraction):
        return p == t
    return str(p) == str(t)


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
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def extract_pred_number(text: str):
    m = re.findall(
        r"answer\s*:\s*([-\$]?\s*\d[\d,]*(?:\.\d+)?(?:\s+\w+)?)",
        text,
        flags=re.I,
    )
    cand = m[-1] if m else None
    if not cand:
        m2 = re.findall(r"[-]?\$?\d[\d,]*(?:\.\d+)?", text)
        cand = m2[-1] if m2 else None
    return _to_number(cand)


def eq(a, b, tol=1e-6):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(1.0, abs(b))


def numbers_equal(pred_text: str, truth_text: str, tol: float = 1e-6) -> bool:
    """Extract numeric value from pred_text and compare to truth_text, numerically with tolerance."""
    a = extract_pred_number(pred_text)
    # allow truth to be either a lone number string ('3/4', '$1,200', '12%', etc.) or a longer text
    b = extract_pred_number(truth_text)
    if b is None:
        b = _to_number(truth_text)
    return eq(a, b, tol)


def is_correct(pred, truth) -> bool:
    """
    Unified answer checker used across all scripts (run_mcts, compare_sbs_mcts, etc.).

    Handles:
      - numeric answers (gsm8k, competition_math, etc.) via numbers_equal
      - MMLU-style multiple choice (A/B/C/D) via answers_equal
      - plain text equality / normalized math fractions via answers_equal
    """
    if truth is None:
        return False

    # First: try numeric comparison (gsm8k, competition_math, etc.).
    # Both arguments as strings, since numbers_equal expects text.
    try:
        if numbers_equal(str(pred), str(truth)):
            return True
    except Exception:
        # If numeric parsing fails for any reason, fall back to text-based comparison
        pass

    # Fallback: MMLU-letter / normalized-answer check
    return answers_equal(pred, truth)
