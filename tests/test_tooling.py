import json
from fractions import Fraction

import pytest

from tooling import TOOL_REGISTRY, ToolExecutionError, evaluate_arithmetic, resolve_tools


def test_evaluate_arithmetic_exactly():
    assert evaluate_arithmetic("(17 * 23) + 1 / 4") == Fraction(1565, 4)
    assert json.loads(TOOL_REGISTRY["calculate"].invoke({"expression": "0.1 + 0.2"}))[
        "result"
    ] == "0.3"


def test_verify_arithmetic_compares_computed_and_claimed_values():
    result = json.loads(
        TOOL_REGISTRY["verify_arithmetic"].invoke(
            {"expression": "12 * 8", "claimed_value": "96"}
        )
    )
    assert result == {
        "expression": "12 * 8",
        "normalized_expression": "12 * 8",
        "computed": "96",
        "claimed_value": "96",
        "matches": True,
    }


def test_tools_normalize_currency_unicode_and_an_equation():
    calculated = json.loads(
        TOOL_REGISTRY["calculate"].invoke({"expression": r"\$1,200 \div 4"})
    )
    assert calculated["result"] == "300"

    verified = json.loads(
        TOOL_REGISTRY["verify_arithmetic"].invoke(
            {"expression": "1/4 + 1/2 = 3/4", "claimed_value": "0.75"}
        )
    )
    assert verified["computed"] == "0.75"
    assert verified["equation_consistent"] is True
    assert verified["matches"] is True


@pytest.mark.parametrize(
    "expression",
    ["sum([1, 2])", "x + 1", "2 ** 101", "1 / 0", "3 == 3"],
)
def test_evaluate_arithmetic_rejects_unsafe_or_unsupported_input(expression):
    with pytest.raises(ToolExecutionError):
        evaluate_arithmetic(expression)


def test_resolve_tools_enforces_the_registry():
    assert [tool.name for tool in resolve_tools(["calculate"])] == ["calculate"]
    with pytest.raises(ValueError, match="Unknown tool"):
        resolve_tools(["not_registered"])
