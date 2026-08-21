import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tooling import ToolSpec

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


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
    tool_names: Dict[str, int] = field(default_factory=dict)

    def add(self, other: "ToolGenerationStats") -> "ToolGenerationStats":
        self.tool_enabled_candidates += other.tool_enabled_candidates
        self.tool_used_candidates += other.tool_used_candidates
        self.model_responses += other.model_responses
        self.tool_calls += other.tool_calls
        self.successful_tool_calls += other.successful_tool_calls
        self.tool_errors += other.tool_errors
        self.invalid_tool_calls += other.invalid_tool_calls
        self.required_tool_misses += other.required_tool_misses
        self.forced_tool_reprompts += other.forced_tool_reprompts
        for name, count in other.tool_names.items():
            self.tool_names[name] = self.tool_names.get(name, 0) + int(count)
        return self

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tool_enabled_candidates": self.tool_enabled_candidates,
            "tool_used_candidates": self.tool_used_candidates,
            "model_responses": self.model_responses,
            "tool_calls": self.tool_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "tool_errors": self.tool_errors,
            "invalid_tool_calls": self.invalid_tool_calls,
            "required_tool_misses": self.required_tool_misses,
            "forced_tool_reprompts": self.forced_tool_reprompts,
            "tool_names": dict(sorted(self.tool_names.items())),
        }


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
        max_tool_calls: int = 4,
        require_tool_call: bool = False,
        expose_tool_trace: bool = True,
    ):
        self.model = model
        self.tok = tok
        self.name = name
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tools = list(tools or [])
        self.max_tool_calls = max(1, int(max_tool_calls))
        self.require_tool_call = bool(require_tool_call and self.tools)
        self.expose_tool_trace = bool(expose_tool_trace)
        self.last_tool_stats = ToolGenerationStats()

    def format_messages(self, user_content: str):
        system_prompt = self.system_prompt
        if self.tools:
            tool_names = ", ".join(tool.name for tool in self.tools)
            requirement = (
                " You must call at least one available tool before giving your final response."
                if self.require_tool_call
                else " Call a tool whenever its result would improve accuracy."
            )
            system_prompt = (
                f"{system_prompt.rstrip()}\n\nAVAILABLE TOOL POLICY: You may use only: "
                f"{tool_names}.{requirement} After receiving tool results, finish in the "
                "output format requested above."
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

    def _tool_schemas(self) -> List[Dict[str, Any]]:
        return [tool.schema for tool in self.tools]

    def _to_inputs(self, msgs, *, tools: Optional[List[Dict[str, Any]]] = None):
        template_kwargs: Dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        if tools:
            template_kwargs["tools"] = tools
        enc = self.tok.apply_chat_template(
            msgs,
            **template_kwargs,
        )

        if isinstance(enc, list):
            return self.tok.pad({"input_ids": [enc]}, return_tensors="pt")

        if isinstance(enc, torch.Tensor):
            if enc.dim() == 1:
                enc = enc.unsqueeze(0)
            return {"input_ids": enc, "attention_mask": torch.ones_like(enc)}

        template_kwargs["tokenize"] = False
        rendered = self.tok.apply_chat_template(msgs, **template_kwargs)
        return self.tok(rendered, return_tensors="pt", add_special_tokens=False)

    @torch.inference_mode()
    def _generate_model_n(
        self,
        msgs,
        n: int,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        **gen_kwargs,
    ) -> List[str]:
        params = self._generation_kwargs(**gen_kwargs)
        inputs = self._to_inputs(msgs, tools=tools)
        inputs = {k: v.to(self.model.device, non_blocking=True) for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            do_sample=True,
            top_p=params["top_p"],
            temperature=params["temperature"],
            num_return_sequences=n,
            max_new_tokens=params["max_new_tokens"],
            use_cache=True,
            pad_token_id=self.tok.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        texts = self.tok.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        return [text.strip() for text in texts]

    def _parse_tool_call(self, text: str) -> Optional[Tuple[str, Dict[str, Any], str]]:
        match = _TOOL_CALL_RE.search(str(text or ""))
        if match is None:
            return None
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
            return None
        arguments = payload.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if not isinstance(arguments, dict):
            return None
        residual = (str(text)[: match.start()] + str(text)[match.end() :]).strip()
        return payload["name"].strip(), arguments, residual

    def _serialize_tool_run(self, trace: List[str], final_text: str) -> str:
        final_text = str(final_text or "").strip()
        if not self.expose_tool_trace or not trace:
            return final_text
        return "\n".join([*trace, "AGENT_RESPONSE:", final_text]).strip()

    def _run_tool_branch(
        self,
        msgs,
        initial_text: str,
        *,
        schemas: List[Dict[str, Any]],
        **gen_kwargs,
    ) -> Tuple[str, ToolGenerationStats]:
        stats = ToolGenerationStats(tool_enabled_candidates=1, model_responses=1)
        available = {tool.name: tool for tool in self.tools}
        working_messages = [dict(message) for message in msgs]
        trace: List[str] = []
        text = str(initial_text or "").strip()
        tool_calls = 0
        required_retries = 0

        while True:
            parsed = self._parse_tool_call(text)
            if parsed is None:
                if "<tool_call>" in text:
                    stats.invalid_tool_calls += 1
                if self.require_tool_call and tool_calls == 0 and required_retries < 2:
                    required_retries += 1
                    stats.forced_tool_reprompts += 1
                    working_messages.extend(
                        [
                            {"role": "assistant", "content": text},
                            {
                                "role": "user",
                                "content": (
                                    "Your response did not call a required tool. Call exactly one "
                                    "available tool now, wait for its result, and only then give "
                                    "the final response."
                                ),
                            },
                        ]
                    )
                    text = self._generate_model_n(
                        working_messages, n=1, tools=schemas, **gen_kwargs
                    )[0]
                    stats.model_responses += 1
                    continue
                if self.require_tool_call and tool_calls == 0:
                    stats.required_tool_misses += 1
                stats.tool_used_candidates += int(tool_calls > 0)
                return self._serialize_tool_run(trace, text), stats

            name, arguments, residual = parsed
            tool_calls += 1
            stats.tool_calls += 1
            stats.tool_names[name] = stats.tool_names.get(name, 0) + 1
            if residual:
                trace.append(f"AGENT_TEXT_BEFORE_TOOL: {residual}")
            trace.append(
                f"TOOL_CALL {name}: {json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"
            )

            tool = available.get(name)
            try:
                if tool is None:
                    raise ValueError(f"Tool {name!r} is not allowed for this agent.")
                result = tool.invoke(arguments)
                stats.successful_tool_calls += 1
            except Exception as exc:
                result = json.dumps({"error": str(exc)}, ensure_ascii=False)
                stats.tool_errors += 1
            trace.append(f"TOOL_RESULT {name}: {result}")

            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": residual,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            }
            working_messages.extend(
                [
                    assistant_message,
                    {"role": "tool", "name": name, "content": result},
                ]
            )

            if tool_calls >= self.max_tool_calls:
                working_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The tool-call budget is exhausted. Give your final response now "
                            "without making another tool call."
                        ),
                    }
                )
                text = self._generate_model_n(working_messages, n=1, tools=None, **gen_kwargs)[0]
                stats.model_responses += 1
                stats.tool_used_candidates += 1
                return self._serialize_tool_run(trace, text), stats

            text = self._generate_model_n(working_messages, n=1, tools=schemas, **gen_kwargs)[0]
            stats.model_responses += 1

    def _generate_local_n(self, msgs, n: int, **gen_kwargs):
        if not self.tools:
            self.last_tool_stats = ToolGenerationStats(model_responses=int(n))
            return self._generate_model_n(msgs, n, **gen_kwargs)

        schemas = self._tool_schemas()
        initial_texts = self._generate_model_n(
            msgs,
            n,
            tools=schemas,
            **gen_kwargs,
        )
        outputs: List[str] = []
        combined_stats = ToolGenerationStats()
        for initial_text in initial_texts:
            output, branch_stats = self._run_tool_branch(
                msgs,
                initial_text,
                schemas=schemas,
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
