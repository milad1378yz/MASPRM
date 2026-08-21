import ast
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional


class ToolExecutionError(ValueError):
    pass


def _bounded_fraction(value: Fraction) -> Fraction:
    if len(str(abs(value.numerator))) > 200 or len(str(value.denominator)) > 200:
        raise ToolExecutionError("Numeric result is too large.")
    return value


def _eval_arithmetic_node(
    node: ast.AST,
    variables: Optional[Mapping[str, Fraction]] = None,
) -> Fraction:
    if isinstance(node, ast.Name) and variables is not None:
        if node.id not in variables:
            raise ToolExecutionError(f"Unknown variable {node.id!r}.")
        return variables[node.id]

    if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
        if isinstance(node.value, int):
            return _bounded_fraction(Fraction(node.value))
        if isinstance(node.value, float) and math.isfinite(node.value):
            return _bounded_fraction(Fraction(str(node.value)))
        raise ToolExecutionError("Only finite numeric literals are allowed.")

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_arithmetic_node(node.operand, variables)
        return value if isinstance(node.op, ast.UAdd) else -value

    if not isinstance(node, ast.BinOp):
        raise ToolExecutionError("Only arithmetic operators are allowed.")

    left = _eval_arithmetic_node(node.left, variables)
    right = _eval_arithmetic_node(node.right, variables)
    try:
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            result = Fraction(left // right)
        elif isinstance(node.op, ast.Mod):
            result = left % right
        elif isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right.numerator) > 100:
                raise ToolExecutionError(
                    "Exponents must be integers with absolute value at most 100."
                )
            result = left**right.numerator
        else:
            raise ToolExecutionError("Unsupported arithmetic operator.")
    except ZeroDivisionError as exc:
        raise ToolExecutionError("Division by zero.") from exc
    return _bounded_fraction(Fraction(result))


def evaluate_arithmetic(expression: str) -> Fraction:
    expression = str(expression).strip()
    if not expression or len(expression) > 256:
        raise ToolExecutionError("Expression must contain between 1 and 256 characters.")
    try:
        parsed = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ToolExecutionError("Expression is not valid arithmetic.") from exc
    return _eval_arithmetic_node(parsed.body)


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)

    denominator = value.denominator
    while denominator % 2 == 0:
        denominator //= 2
    while denominator % 5 == 0:
        denominator //= 5
    if denominator == 1:
        rendered = f"{float(value):.12f}".rstrip("0").rstrip(".")
        return rendered or "0"
    return f"{value.numerator}/{value.denominator}"


def _parse_claimed_number(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise ToolExecutionError("Claimed value must be numeric.")
    if isinstance(value, (int, float)):
        return evaluate_arithmetic(str(value))

    text = str(value).strip().replace(",", "")
    if text.startswith("$"):
        text = text[1:].strip()
    if text.endswith("%"):
        return evaluate_arithmetic(text[:-1].strip()) / 100
    return evaluate_arithmetic(text)


def _evaluate_tool_expression(expression: str) -> tuple[str, Fraction, List[Fraction]]:
    """Tolerate harmless calculator notation while keeping the AST evaluator strict."""
    normalized = (
        expression.strip()
        .replace("\\$", "")
        .replace("$", "")
        .replace(",", "")
        .replace("\\times", "*")
        .replace("\\cdot", "*")
        .replace("\\div", "/")
        .replace("\\(", "")
        .replace("\\)", "")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("^", "**")
    )
    if "==" in normalized or "!=" in normalized or "<=" in normalized or ">=" in normalized:
        raise ToolExecutionError("Only a single arithmetic expression or equation is allowed.")
    parts = [part.strip() for part in normalized.split("=")]
    if not parts or any(not part for part in parts):
        raise ToolExecutionError("Expression is not valid arithmetic.")
    values = [evaluate_arithmetic(part) for part in parts]
    return normalized, values[0], values


def _calculate(arguments: Mapping[str, Any]) -> str:
    expression = arguments.get("expression")
    if not isinstance(expression, str):
        raise ToolExecutionError("'expression' must be a string.")
    normalized, result, equation_values = _evaluate_tool_expression(expression)
    payload: Dict[str, Any] = {
        "expression": expression,
        "normalized_expression": normalized,
        "result": _format_fraction(result),
    }
    if len(equation_values) > 1:
        payload["equation_values"] = [_format_fraction(value) for value in equation_values]
        payload["equation_consistent"] = len(set(equation_values)) == 1
    return json.dumps(payload, ensure_ascii=False)


def _verify_arithmetic(arguments: Mapping[str, Any]) -> str:
    expression = arguments.get("expression")
    if not isinstance(expression, str):
        raise ToolExecutionError("'expression' must be a string.")
    if "claimed_value" not in arguments:
        raise ToolExecutionError("'claimed_value' is required.")

    normalized, computed, equation_values = _evaluate_tool_expression(expression)
    claimed = _parse_claimed_number(arguments["claimed_value"])
    payload: Dict[str, Any] = {
        "expression": expression,
        "normalized_expression": normalized,
        "computed": _format_fraction(computed),
        "claimed_value": _format_fraction(claimed),
        "matches": computed == claimed,
    }
    if len(equation_values) > 1:
        payload["equation_values"] = [_format_fraction(value) for value in equation_values]
        payload["equation_consistent"] = len(set(equation_values)) == 1
    return json.dumps(payload, ensure_ascii=False)


def _run_numeric_python(arguments: Mapping[str, Any]) -> str:
    """Run a tiny, deterministic Python subset for numeric scratch work.

    Model-produced code is untrusted, so this deliberately interprets only numeric
    assignments and ``print(...)`` instead of passing text to Python's ``exec``.
    """
    code = arguments.get("code")
    if not isinstance(code, str):
        raise ToolExecutionError("'code' must be a string.")
    code = code.strip()
    if not code or len(code) > 512:
        raise ToolExecutionError("Code must contain between 1 and 512 characters.")
    try:
        parsed = ast.parse(code, mode="exec")
    except (SyntaxError, ValueError) as exc:
        raise ToolExecutionError("Code is not valid Python.") from exc
    if len(parsed.body) > 32:
        raise ToolExecutionError("Code may contain at most 32 statements.")

    variables: Dict[str, Fraction] = {}
    stdout: List[str] = []
    last_result: Optional[Fraction] = None
    for statement in parsed.body:
        if isinstance(statement, ast.Assign):
            if (
                len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)
                or statement.targets[0].id.startswith("_")
            ):
                raise ToolExecutionError("Assignments require one simple variable name.")
            variables[statement.targets[0].id] = _eval_arithmetic_node(
                statement.value, variables
            )
            continue

        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Name)
            and statement.value.func.id == "print"
        ):
            call = statement.value
            if not call.args or call.keywords:
                raise ToolExecutionError("print requires numeric positional arguments.")
            values = [_eval_arithmetic_node(arg, variables) for arg in call.args]
            stdout.append(" ".join(_format_fraction(value) for value in values))
            last_result = values[-1]
            continue

        raise ToolExecutionError(
            "Only numeric assignments and print(arithmetic) are allowed; imports, "
            "loops, functions, attributes, and file access are disabled."
        )

    if last_result is None:
        raise ToolExecutionError("Code must print the final numeric answer.")
    return json.dumps(
        {"stdout": "\n".join(stdout), "result": _format_fraction(last_result)},
        ensure_ascii=False,
    )


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[[Mapping[str, Any]], str]

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def invoke(self, arguments: Mapping[str, Any]) -> str:
        if not isinstance(arguments, Mapping):
            raise ToolExecutionError("Tool arguments must be a JSON object.")
        return self.handler(arguments)


TOOL_REGISTRY: Dict[str, ToolSpec] = {
    "solver_python": ToolSpec(
        name="solver_python",
        description=(
            "Run a restricted numeric Python scratchpad. Use numeric assignments and "
            "print only; for example: total = 12 * 3; print(total)."
        ),
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
        handler=_run_numeric_python,
    ),
    "verifier_python": ToolSpec(
        name="verifier_python",
        description=(
            "Run a restricted numeric Python scratchpad to independently verify an "
            "answer. Use numeric assignments and print only."
        ),
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
        handler=_run_numeric_python,
    ),
    "calculate": ToolSpec(
        name="calculate",
        description=(
            "Evaluate one arithmetic expression exactly. Use only numeric literals, "
            "parentheses, and +, -, *, /, //, %, or ** operators."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The complete arithmetic expression to evaluate.",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        handler=_calculate,
    ),
    "verify_arithmetic": ToolSpec(
        name="verify_arithmetic",
        description=(
            "Independently evaluate an arithmetic expression and report whether it "
            "matches a claimed numeric value."
        ),
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression that should produce the claim.",
                },
                "claimed_value": {
                    "description": "The numeric result claimed by the proposed solution.",
                    "anyOf": [{"type": "number"}, {"type": "string"}],
                },
            },
            "required": ["expression", "claimed_value"],
            "additionalProperties": False,
        },
        handler=_verify_arithmetic,
    ),
}


def resolve_tools(names: Iterable[str]) -> List[ToolSpec]:
    tools: List[ToolSpec] = []
    for raw_name in names:
        name = str(raw_name).strip()
        if name not in TOOL_REGISTRY:
            available = ", ".join(sorted(TOOL_REGISTRY))
            raise ValueError(f"Unknown tool {name!r}. Available tools: {available}.")
        tools.append(TOOL_REGISTRY[name])
    return tools
