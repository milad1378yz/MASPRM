import re
from fractions import Fraction
from typing import Optional


FINAL_SPAN_RE = re.compile(
    r"\b(?:final\s+answer|answer)\s*:?\s*(.+)",
    re.IGNORECASE,
)


def extract_final_span(text: str) -> Optional[str]:
    """Return the last explicitly tagged answer span, if present."""
    matches = FINAL_SPAN_RE.findall(str(text or ""))
    return matches[-1].strip() if matches else None


def extract_last_boxed(text: str) -> Optional[str]:
    r"""Extract the last balanced ``\boxed{...}`` payload, including nested braces."""
    source = str(text or "")
    result = None
    for match in re.finditer(r"\\boxed\s*\{", source):
        start = match.end()
        depth = 1
        for idx in range(start, len(source)):
            if source[idx] == "{":
                depth += 1
            elif source[idx] == "}":
                depth -= 1
                if depth == 0:
                    result = source[start:idx]
                    break
    return result


def normalize_answer(ans: str):
    explicit = extract_final_span(ans)
    ans = explicit if explicit is not None else str(ans)
    boxed = extract_last_boxed(ans)
    if boxed is not None:
        ans = boxed
    ans = re.sub(r"\\frac\s*\{([-+]?\d+)\}\s*\{([-+]?\d+)\}", r"\1/\2", ans)
    ans = re.sub(r"\\text\{[^{}]*\}\s*$", "", ans)
    ans = ans.replace("\\left", "").replace("\\right", "")
    ans = ans.replace("$", "").replace("\\(", "").replace("\\)", "")
    ans = ans.strip().strip("*`.,")
    number = _to_number(ans)
    if number is not None:
        return number
    return re.sub(r"[\s,]+", "", ans).lower()


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
    s = str(s).strip().strip("*`.,")
    boxed = extract_last_boxed(s)
    if boxed is not None:
        s = boxed
    s = re.sub(r"\\frac\s*\{([-+]?\d+)\}\s*\{([-+]?\d+)\}", r"\1/\2", s)
    s = re.sub(r"\\text\{[^{}]*\}\s*$", "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\(", "").replace("\\)", "").strip()
    s = s.replace(",", "")
    s = re.sub(r"^\$", "", s)
    s = re.sub(
        r"\s*(?:dollars?|tickets?|units?|boxes?|people|students?|cm|mm|km|m|degrees?)"
        r"(?:\s*\^?\s*[23]|[²³])?$",
        "",
        s,
        flags=re.I,
    )
    if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", s):
        return float(Fraction(s.replace(" ", "")))
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
    if "TOOL_STATUS:" in text:
        status = text.rsplit("TOOL_STATUS:", 1)[-1].strip().splitlines()[0]
        if status.lower() != "success":
            return None
    if "TERMINAL_LOG:" in text:
        text = text.rsplit("TERMINAL_LOG:", 1)[-1]
    explicit = extract_final_span(text)
    source = explicit if explicit is not None else text
    direct = _to_number(source)
    if direct is not None:
        return direct
    latex_fractions = re.findall(
        r"\\frac\s*\{([-+]?\d+)\}\s*\{([-+]?\d+)\}", source
    )
    if latex_fractions:
        numerator, denominator = latex_fractions[-1]
        return _to_number(f"{numerator}/{denominator}")
    fractions = re.findall(r"[-+]?\d+\s*/\s*[-+]?\d+", source)
    if fractions:
        return _to_number(fractions[-1])
    tokens = re.findall(
        r"[-+]?\$?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?%?",
        source,
    )
    return _to_number(tokens[-1]) if tokens else None


def eq(a, b, tol=1e-3):
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def numbers_equal(pred_text: str, truth_text: str, tol: float = 1e-3) -> bool:
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
    pred = str(pred)
    if "TOOL_STATUS:" in pred:
        status = pred.rsplit("TOOL_STATUS:", 1)[-1].strip().splitlines()[0]
        if status.lower() != "success":
            return False
        if "TERMINAL_LOG:" in pred:
            pred = pred.rsplit("TERMINAL_LOG:", 1)[-1].split("TOOL_STATUS:", 1)[0]

    # Numeric datasets must be decided by the stated final number. Falling through
    # to text normalization can accidentally match an earlier number in the work.
    truth_number = _to_number(str(truth))
    if truth_number is not None:
        return eq(extract_pred_number(pred), truth_number)

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
