import re
from dataclasses import dataclass
from typing import Iterable, List

from smolagents import LocalPythonExecutor


class ToolExecutionError(ValueError):
    pass


_NUMBER_RE = re.compile(
    r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)|(?:[-+]?\d+/\d+)"
)


@dataclass(frozen=True)
class PythonExecution:
    code: str
    logs: str
    result: str


@dataclass(frozen=True)
class ToolSpec:
    name: str = "python"

    def invoke(self, code: str) -> PythonExecution:
        code = str(code or "").strip()
        if not code or len(code) > 1024:
            raise ToolExecutionError("Python code must contain 1 to 1024 characters.")

        executor = LocalPythonExecutor(
            additional_authorized_imports=[],
            max_print_outputs_length=2000,
            timeout_seconds=2,
        )
        executor.send_tools({})
        try:
            output = executor(code)
        except Exception as exc:
            raise ToolExecutionError(str(exc)) from exc

        logs = str(output.logs or "")
        if not logs.strip() and output.output is not None:
            logs = f"{output.output}\n"
        lines = [line.strip() for line in logs.splitlines() if line.strip()]
        if not lines:
            raise ToolExecutionError("Python must output the final numeric answer.")

        result = lines[-1].replace(",", "").lstrip("$").rstrip("%")
        if _NUMBER_RE.fullmatch(result) is None:
            raise ToolExecutionError("The final execution-log line must be only a number.")
        return PythonExecution(code=code, logs=logs, result=result)


PYTHON_TOOL = ToolSpec()


def resolve_tools(names: Iterable[str]) -> List[ToolSpec]:
    requested = list(dict.fromkeys(str(name).strip() for name in names))
    if not requested:
        return []
    if requested != [PYTHON_TOOL.name]:
        raise ValueError("The only available tool is 'python'.")
    return [PYTHON_TOOL]
