from typing import Optional

import requests
import torch
from openai import OpenAI
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
        # openai parameters
        use_openai: bool = False,
        openai_model: str = "gpt-4.1-mini",
        openai_client: Optional[OpenAI] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        # runpod parameters
        use_runpod: bool = False,
    ):
        """
        Supports Local (transformers), OpenAI, and RunPod Serverless backends.
        """
        self.model = model
        self.tok = tok
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.use_openai = use_openai
        self.use_runpod = use_runpod
        self.openai_model = openai_model
        self.runpod_base_url = openai_base_url
        self.runpod_api_key = openai_api_key
        self.runpod_model = openai_model

        if use_openai and openai_client is None:
            openai_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
        self.client = openai_client if use_openai else None

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

    def _normalize_messages(self, msgs):
        if not msgs:
            return [{"role": "system", "content": self.system_prompt}]
        if msgs[0].get("role") != "system":
            return [{"role": "system", "content": self.system_prompt}] + msgs
        return msgs

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

    def _generate_openai_n(self, msgs, n: int, **gen_kwargs):
        params = self._generation_kwargs(**gen_kwargs)
        messages = self._normalize_messages(msgs)

        try:
            resp = self.client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=params["temperature"],
                top_p=params["top_p"],
                max_tokens=params["max_new_tokens"],
                n=n,
            )
            return [choice.message.content.strip() for choice in resp.choices]
        except Exception:
            if n <= 1:
                raise
            return [self._generate_openai_n(msgs, 1, **gen_kwargs)[0] for _ in range(n)]

    def _generate_runpod_once(self, msgs, **gen_kwargs) -> str:
        params = self._generation_kwargs(**gen_kwargs)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.runpod_api_key}",
        }
        data = {
            "input": {
                "openai_route": "/v1/chat/completions",
                "openai_input": {
                    "model": self.runpod_model,
                    "messages": self._normalize_messages(msgs),
                    "max_tokens": params["max_new_tokens"],
                    "temperature": params["temperature"],
                    "top_p": params["top_p"],
                },
            }
        }

        try:
            response = requests.post(self.runpod_base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            output = result.get("output", {})
            return output[0]["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            return f"RunPod Error: {e}"
        except (KeyError, IndexError) as e:
            return f"RunPod Parse Error: {e} - Raw: {result}"

    def _generate_runpod_n(self, msgs, n: int, **gen_kwargs):
        return [self._generate_runpod_once(msgs, **gen_kwargs) for _ in range(n)]

    def generate(self, msgs, **gen_kwargs) -> str:
        return self.generate_n(msgs, n=1, **gen_kwargs)[0]

    def generate_n(self, msgs, n: int, **gen_kwargs):
        if self.use_openai:
            return self._generate_openai_n(msgs, n, **gen_kwargs)
        if self.use_runpod:
            return self._generate_runpod_n(msgs, n, **gen_kwargs)
        return self._generate_local_n(msgs, n, **gen_kwargs)
