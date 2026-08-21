import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from smolagents import parse_code_blobs
from transformers import AutoModelForCausalLM, AutoTokenizer

from tooling import ToolSpec

_PYTHON_BLOCK_RE = re.compile(r"```(?:python|py)\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class ToolGenerationStats:
    tool_enabled_candidates: int = 0
    tool_used_candidates: int = 0
    model_responses: int = 0
    tool_calls: int = 0
    successful_tool_calls: int = 0
    tool_errors: int = 0
    invalid_tool_calls: int = 0
    required_tool_misses: int = 0
    forced_tool_reprompts: int = 0
    model_prompt_tokens: int = 0
    max_prompt_tokens: int = 0
    truncated_responses: int = 0
    tool_names: Dict[str, int] = field(default_factory=dict)

    def record_generation(
        self, prompt_tokens: int, truncated: Sequence[bool]
    ) -> "ToolGenerationStats":
        count = len(truncated)
        self.model_responses += count
        self.model_prompt_tokens += int(prompt_tokens) * count
        if count:
            self.max_prompt_tokens = max(self.max_prompt_tokens, int(prompt_tokens))
        self.truncated_responses += sum(bool(value) for value in truncated)
        return self

    def add(self, other: "ToolGenerationStats") -> "ToolGenerationStats":
        for field_name in (
            "tool_enabled_candidates",
            "tool_used_candidates",
            "model_responses",
            "tool_calls",
            "successful_tool_calls",
            "tool_errors",
            "invalid_tool_calls",
            "required_tool_misses",
            "forced_tool_reprompts",
            "model_prompt_tokens",
            "truncated_responses",
        ):
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))
        self.max_prompt_tokens = max(self.max_prompt_tokens, other.max_prompt_tokens)
        for name, count in other.tool_names.items():
            self.tool_names[name] = self.tool_names.get(name, 0) + int(count)
        return self


class Agent:
    def __init__(
        self,
        model: Optional[AutoModelForCausalLM] = None,
        tok: Optional[AutoTokenizer] = None,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        name: Optional[str] = None,
        tools: Optional[Sequence[ToolSpec]] = None,
        max_tool_calls: int = 3,
    ):
        self.model = model
        self.tok = tok
        self.name = name
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tools = list(tools or [])
        if len(self.tools) > 1 or (self.tools and self.tools[0].name != "python"):
            raise ValueError("An agent may use only the single 'python' tool.")
        self.python_tool = self.tools[0] if self.tools else None
        self.max_tool_calls = max(1, int(max_tool_calls))
        self.last_tool_stats = ToolGenerationStats()
        self._last_generation_prompt_tokens = 0
        self._last_generation_truncated: List[bool] = []

    def format_messages(self, user_content: str):
        system_prompt = self.system_prompt.rstrip()
        if self.python_tool is not None:
            system_prompt += (
                "\nReturn one short ```python code block with no imports. Its last "
                "expression or printed line must be the requested number."
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _generation_kwargs(self, **gen_kwargs):
        global_cap = int(gen_kwargs.get("max_new_tokens", self.max_new_tokens))
        return {
            "temperature": gen_kwargs.get("temperature", self.temperature),
            "top_p": gen_kwargs.get("top_p", self.top_p),
            "max_new_tokens": min(global_cap, self.max_new_tokens),
        }

    def _to_inputs(self, msgs):
        enc = self.tok.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(enc, list):
            return self.tok.pad({"input_ids": [enc]}, return_tensors="pt")
        if isinstance(enc, torch.Tensor):
            if enc.dim() == 1:
                enc = enc.unsqueeze(0)
            return {"input_ids": enc, "attention_mask": torch.ones_like(enc)}
        rendered = self.tok.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tok(rendered, return_tensors="pt", add_special_tokens=False)

    @torch.inference_mode()
    def _generate_model_n(self, msgs, n: int, **gen_kwargs) -> List[str]:
        params = self._generation_kwargs(**gen_kwargs)
        inputs = self._to_inputs(msgs)
        inputs = {key: value.to(self.model.device, non_blocking=True) for key, value in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            do_sample=True,
            top_p=params["top_p"],
            temperature=params["temperature"],
            num_return_sequences=n,
            max_new_tokens=params["max_new_tokens"],
            use_cache=True,
            pad_token_id=self.tok.eos_token_id,
        )

        prompt_tokens = int(inputs["input_ids"].shape[1])
        generated_ids = output_ids[:, prompt_tokens:]
        eos_ids = {int(self.tok.eos_token_id)} if self.tok.eos_token_id is not None else set()
        configured_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(configured_eos, (list, tuple, set)):
            eos_ids.update(int(token_id) for token_id in configured_eos)
        elif configured_eos is not None:
            eos_ids.add(int(configured_eos))
        hit_cap = generated_ids.shape[1] >= params["max_new_tokens"]
        self._last_generation_prompt_tokens = prompt_tokens
        self._last_generation_truncated = [
            bool(hit_cap and not any(int(token_id) in eos_ids for token_id in row.tolist()))
            for row in generated_ids
        ]
        texts = self.tok.batch_decode(generated_ids, skip_special_tokens=True)
        return [text.strip() for text in texts]

    def _parse_python(self, text: str) -> Optional[Tuple[str, str]]:
        try:
            code = parse_code_blobs(
                str(text or ""),
                code_block_tags=(r"```(?:python|py)", "\n```"),
            ).strip()
        except ValueError:
            return None
        match = _PYTHON_BLOCK_RE.search(str(text or ""))
        residual = (
            (str(text)[: match.start()] + str(text)[match.end() :]).strip()
            if match is not None
            else ""
        )
        return code, residual

    def _run_python_branch(
        self,
        msgs,
        initial_text: str,
        initial_prompt_tokens: int,
        initial_truncated: bool,
        **gen_kwargs,
    ) -> Tuple[str, ToolGenerationStats]:
        stats = ToolGenerationStats(tool_enabled_candidates=1)
        stats.record_generation(initial_prompt_tokens, [initial_truncated])
        working_messages = [dict(message) for message in msgs]
        text = str(initial_text or "").strip()
        trace: List[str] = []

        for attempt in range(self.max_tool_calls):
            parsed = self._parse_python(text)
            if parsed is None:
                stats.invalid_tool_calls += 1
                error = "No valid Python code block was found."
            else:
                code, residual = parsed
                if residual:
                    trace.append(f"AGENT_TEXT_BEFORE_PYTHON: {residual}")
                trace.append(f"PYTHON_CODE:\n{code}")
                stats.tool_calls += 1
                stats.tool_names["python"] = stats.tool_names.get("python", 0) + 1
                try:
                    execution = self.python_tool.invoke(code)
                except Exception as exc:
                    stats.tool_errors += 1
                    error = str(exc)
                    trace.append(f"EXECUTION_ERROR:\n{error}")
                else:
                    stats.successful_tool_calls += 1
                    stats.tool_used_candidates += 1
                    trace.append(f"EXECUTION_LOGS:\n{execution.logs.rstrip()}")
                    return (
                        "\n".join([*trace, "AGENT_RESPONSE:", f"Final Answer: {execution.result}"]),
                        stats,
                    )

            if attempt + 1 >= self.max_tool_calls:
                stats.required_tool_misses += 1
                return "\n".join([*trace, "AGENT_RESPONSE:", text]).strip(), stats

            stats.forced_tool_reprompts += 1
            working_messages.extend(
                [
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            f"Python execution failed: {error[:1200]}\n"
                            "Return corrected Python that outputs only the requested number."
                        ),
                    },
                ]
            )
            text = self._generate_model_n(working_messages, n=1, **gen_kwargs)[0]
            stats.record_generation(
                self._last_generation_prompt_tokens,
                self._last_generation_truncated,
            )

        raise AssertionError("unreachable")

    def _generate_local_n(self, msgs, n: int, **gen_kwargs):
        initial_texts = self._generate_model_n(msgs, n, **gen_kwargs)
        initial_prompt_tokens = self._last_generation_prompt_tokens
        initial_truncated = list(self._last_generation_truncated)
        if self.python_tool is None:
            self.last_tool_stats = ToolGenerationStats().record_generation(
                initial_prompt_tokens,
                initial_truncated,
            )
            return initial_texts

        outputs: List[str] = []
        combined_stats = ToolGenerationStats()
        for index, initial_text in enumerate(initial_texts):
            output, branch_stats = self._run_python_branch(
                msgs,
                initial_text,
                initial_prompt_tokens,
                initial_truncated[index],
                **gen_kwargs,
            )
            outputs.append(output)
            combined_stats.add(branch_stats)
        self.last_tool_stats = combined_stats
        return outputs

    def generate(self, msgs, **gen_kwargs) -> str:
        return self.generate_n(msgs, n=1, **gen_kwargs)[0]

    def generate_n(self, msgs, n: int, **gen_kwargs):
        return self._generate_local_n(msgs, n, **gen_kwargs)
