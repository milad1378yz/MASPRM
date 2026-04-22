from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Agent:
    def __init__(
        self,
        model: Optional[AutoModelForCausalLM] = None,
        tok: Optional[AutoTokenizer] = None,
        system_prompt: str = "You are a helpful assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        self.model = model
        self.tok = tok
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def format_messages(self, user_content: str):
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _generation_kwargs(self, **gen_kwargs):
        return {
            "temperature": gen_kwargs.get("temperature", self.temperature),
            "top_p": gen_kwargs.get("top_p", self.top_p),
            "max_new_tokens": gen_kwargs.get("max_new_tokens", self.max_new_tokens),
        }

    def _to_inputs(self, msgs):
        enc = self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)

        if isinstance(enc, list):
            return self.tok.pad({"input_ids": [enc]}, return_tensors="pt")

        if isinstance(enc, torch.Tensor):
            if enc.dim() == 1:
                enc = enc.unsqueeze(0)
            return {"input_ids": enc, "attention_mask": torch.ones_like(enc)}

        rendered = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(rendered, return_tensors="pt", add_special_tokens=False)

    @torch.inference_mode()
    def _generate_local_n(self, msgs, n: int, **gen_kwargs):
        params = self._generation_kwargs(**gen_kwargs)
        inputs = self._to_inputs(msgs)
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

    def generate(self, msgs, **gen_kwargs) -> str:
        return self.generate_n(msgs, n=1, **gen_kwargs)[0]

    def generate_n(self, msgs, n: int, **gen_kwargs):
        return self._generate_local_n(msgs, n, **gen_kwargs)
