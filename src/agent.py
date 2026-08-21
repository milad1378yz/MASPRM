from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tooling import ToolSpec


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
        self.last_tool_stats = ToolGenerationStats()
        self._last_generation_prompt_tokens = 0
        self._last_generation_truncated: List[bool] = []

    def format_messages(self, user_content: str):
        system_prompt = self.system_prompt.rstrip()
        if self.python_tool is not None:
            system_prompt += (
                "\nReturn concise executable Python without Markdown fences or prose; "
                "prefer one `print(numeric_expression)` line. Printed output becomes "
                "the terminal log."
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

    def _generate_local_n(self, msgs, n: int, **gen_kwargs):
        texts = self._generate_model_n(msgs, n, **gen_kwargs)
        stats = ToolGenerationStats().record_generation(
            self._last_generation_prompt_tokens,
            self._last_generation_truncated,
        )
        if self.python_tool is None:
            self.last_tool_stats = stats
            return texts

        outputs: List[str] = []
        stats.tool_enabled_candidates = len(texts)
        for text in texts:
            trace: List[str] = []
            for attempt in range(2):
                execution = self.python_tool.invoke(text)
                stats.tool_calls += 1
                stats.tool_names["python"] = stats.tool_names.get("python", 0) + 1
                trace.append(
                    f"PYTHON_CODE:\n{execution.code}\n"
                    f"TERMINAL_LOG:\n{execution.logs.rstrip()}\n"
                    f"TOOL_STATUS: {'success' if execution.succeeded else 'error'}"
                )
                if execution.succeeded:
                    stats.successful_tool_calls += 1
                    stats.tool_used_candidates += 1
                    break

                stats.tool_errors += 1
                if attempt == 0:
                    stats.forced_tool_reprompts += 1
                    retry_messages = [
                        *msgs,
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": (
                                f"TERMINAL_LOG:\n{execution.logs.rstrip()}\n"
                                "Return corrected raw Python only."
                            ),
                        },
                    ]
                    text = self._generate_model_n(retry_messages, 1, **gen_kwargs)[0]
                    stats.record_generation(
                        self._last_generation_prompt_tokens,
                        self._last_generation_truncated,
                    )
            if not execution.succeeded:
                stats.required_tool_misses += 1
            outputs.append("\n".join(trace))
        self.last_tool_stats = stats
        return outputs

    def generate(self, msgs, **gen_kwargs) -> str:
        return self.generate_n(msgs, n=1, **gen_kwargs)[0]

    def generate_n(self, msgs, n: int, **gen_kwargs):
        return self._generate_local_n(msgs, n, **gen_kwargs)
