import re
from dataclasses import dataclass
from threading import Lock
from typing import Iterable, List

from pydantic_monty import CollectString, Monty

_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)?[ \t]*\r?\n(?P<code>.*?)(?:\r?\n)?```",
    re.DOTALL | re.IGNORECASE,
)
_INLINE_CODE_RE = re.compile(r"^`(?P<code>[^`]*)`$", re.DOTALL)
_MAX_CODE_LENGTH = 4096
_MAX_LOG_BYTES = 2000
_MAX_MEMORY_BYTES = 64 * 1024 * 1024
_TIMEOUT_SECONDS = 2
_EXECUTION_LOCK = Lock()


@dataclass(frozen=True)
class PythonExecution:
    code: str
    logs: str
    succeeded: bool


@dataclass(frozen=True)
class ToolSpec:
    name: str = "python"

    def invoke(self, code: str) -> PythonExecution:
        code = str(code or "").strip()
        match = _CODE_BLOCK_RE.search(code)
        if match is None:
            match = _INLINE_CODE_RE.fullmatch(code)
        if match is not None:
            code = match.group("code").strip()
        code = code.replace("\\_", "_")
        if not code or len(code) > _MAX_CODE_LENGTH:
            return PythonExecution(
                code=code,
                logs=f"Python code must contain 1 to {_MAX_CODE_LENGTH} characters.\n",
                succeeded=False,
            )

        terminal = CollectString(max_bytes=_MAX_LOG_BYTES)
        try:
            with _EXECUTION_LOCK:
                with Monty(request_timeout=_TIMEOUT_SECONDS + 1) as pool:
                    with pool.checkout(
                        limits={
                            "max_duration_secs": _TIMEOUT_SECONDS,
                            "max_memory": _MAX_MEMORY_BYTES,
                        }
                    ) as session:
                        result = session.feed_run(code, print_callback=terminal)
        except Exception as exc:
            display = getattr(exc, "display", None)
            error = str(display() if callable(display) else exc)
            return PythonExecution(
                code=code,
                logs=f"{terminal.output}{error.rstrip()}\n",
                succeeded=False,
            )

        logs = terminal.output
        if result is not None:
            logs += f"{result}\n"
        if not logs.strip():
            return PythonExecution(
                code=code,
                logs="Python ran but produced no terminal output. Add print(...).\n",
                succeeded=False,
            )
        return PythonExecution(code=code, logs=logs, succeeded=True)


PYTHON_TOOL = ToolSpec()


def resolve_tools(names: Iterable[str]) -> List[ToolSpec]:
    requested = list(dict.fromkeys(str(name).strip() for name in names))
    if not requested:
        return []
    if requested != [PYTHON_TOOL.name]:
        raise ValueError("The only available tool is 'python'.")
    return [PYTHON_TOOL]
