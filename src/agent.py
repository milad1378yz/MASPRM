import torch
import requests
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from openai import OpenAI



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

        # --- OpenAI Setup ---
        self.use_openai = use_openai
        self.openai_model = openai_model
        if use_openai:
            if openai_client is not None:
                self.client = openai_client
            else:
                self.client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
        else:
            self.client = None

        # --- RunPod Setup ---
        self.use_runpod = use_runpod
        self.runpod_base_url = openai_base_url
        self.runpod_api_key = openai_api_key
        self.runpod_model = openai_model

    # ========= LOCAL (transformers) BACKEND ========= #

    def _to_inputs(self, msgs):
        enc = self.tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True
        )

        if isinstance(enc, list):  # list[int]
            return self.tok.pad({"input_ids": [enc]}, return_tensors="pt")

        if isinstance(enc, torch.Tensor):  # 1D or 2D tensor of ids
            if enc.dim() == 1:
                enc = enc.unsqueeze(0)  # [1, L]
            attn = torch.ones_like(enc)
            return {"input_ids": enc, "attention_mask": attn}

        # Fallback: build string and tokenize normally (slower but safe)
        s = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return self.tok(s, return_tensors="pt", add_special_tokens=False)

    @torch.inference_mode()
    def _generate_local(self, msgs, **gen_kwargs) -> str:
        inputs = self._to_inputs(msgs)
        inputs = {
            k: v.to(self.model.device, non_blocking=True)
            for k, v in inputs.items()
        }

        out = self.model.generate(
            **inputs,
            do_sample=True,
            top_p=gen_kwargs.get("top_p", self.top_p),
            temperature=gen_kwargs.get("temperature", self.temperature),
            max_new_tokens=gen_kwargs.get("max_new_tokens", self.max_new_tokens),
            use_cache=True,
            pad_token_id=self.tok.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        text = self.tok.decode(out[0, prompt_len:], skip_special_tokens=True)
        return text.strip()

    @torch.inference_mode()
    def _generate_n_local(self, msgs, n: int, **gen_kwargs):
        inputs = self._to_inputs(msgs)
        inputs = {
            k: v.to(self.model.device, non_blocking=True)
            for k, v in inputs.items()
        }

        out = self.model.generate(
            **inputs,
            do_sample=True,
            top_p=gen_kwargs.get("top_p", self.top_p),
            temperature=gen_kwargs.get("temperature", self.temperature),
            num_return_sequences=n,
            max_new_tokens=gen_kwargs.get("max_new_tokens", self.max_new_tokens),
            use_cache=True,
            pad_token_id=self.tok.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        texts = self.tok.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        return [t.strip() for t in texts]

    # ========= OPENAI BACKEND ========= #

    def _openai_messages(self, msgs):
        """
        Ensure we always send a system message + user/assistant msgs
        in OpenAI's format: [{"role": "...", "content": "..."}].
        """
        if not msgs:
            return [{"role": "system", "content": self.system_prompt}]

        # If user didn't explicitly provide a system msg, prepend one
        if msgs[0].get("role") != "system":
            msgs = [{"role": "system", "content": self.system_prompt}] + msgs
        return msgs

    def _generate_openai(self, msgs, **gen_kwargs) -> str:
        messages = self._openai_messages(msgs)

        resp = self.client.chat.completions.create(
            model=self.openai_model,
            messages=messages,
            temperature=gen_kwargs.get("temperature", self.temperature),
            top_p=gen_kwargs.get("top_p", self.top_p),
            max_tokens=gen_kwargs.get("max_new_tokens", self.max_new_tokens),
            n=1,
        )
        return resp.choices[0].message.content.strip()

    def _generate_n_openai(self, msgs, n: int, **gen_kwargs):
        messages = self._openai_messages(msgs)
        try:
            resp = self.client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=gen_kwargs.get("temperature", self.temperature),
                top_p=gen_kwargs.get("top_p", self.top_p),
                max_tokens=gen_kwargs.get("max_new_tokens", self.max_new_tokens),
                n=n,
            )
            return [c.message.content.strip() for c in resp.choices]
        except Exception:
            return [self._generate_openai(msgs, **gen_kwargs) for _ in range(n)]

    # ========= RUNPOD BACKEND (New) ========= #

    def _generate_runpod(self, msgs, **gen_kwargs) -> str:

        url = self.runpod_base_url
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.runpod_api_key}",
        }
        
        # Reuse helper to format messages correctly
        messages = self._openai_messages(msgs)

        data = {
            "input": {
                "openai_route": "/v1/chat/completions",
                "openai_input": {
                    "model": self.runpod_model,
                    "messages": messages,
                    "max_tokens": gen_kwargs.get("max_new_tokens", self.max_new_tokens),
                    "temperature": gen_kwargs.get("temperature", self.temperature),
                    "top_p": gen_kwargs.get("top_p", self.top_p),
                }
            }
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            # RunPod 'runsync' usually wraps the result in 'output'
            # Since we hit /v1/chat/completions, 'output' should mimic OpenAI response
            output = result.get("output", {})
            return output[0]["choices"][0]["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            return f"RunPod Error: {e}"
        except (KeyError, IndexError) as e:
            return f"RunPod Parse Error: {e} - Raw: {result}"

    def _generate_n_runpod(self, msgs, n: int, **gen_kwargs):
        # RunPod sync usually returns one response per request.
        # We loop here to ensure stability.
        return [self._generate_runpod(msgs, **gen_kwargs) for _ in range(n)]

    # ========= PUBLIC API ========= #

    def generate(self, msgs, **gen_kwargs) -> str:
        """
        Public method: same signature as before.
        Routes to local HF model (default) or OpenAI depending on `use_openai`.
        """
        if self.use_openai:
            return self._generate_openai(msgs, **gen_kwargs)
        elif self.use_runpod: 
            return self._generate_runpod(msgs, **gen_kwargs)
        else:
            return self._generate_local(msgs, **gen_kwargs)

    def generate_n(self, msgs, n: int, **gen_kwargs):
        if self.use_openai:
            return self._generate_n_openai(msgs, n, **gen_kwargs)
        elif self.use_runpod: 
            return self._generate_n_runpod(msgs, n, **gen_kwargs)
        else:
            return self._generate_n_local(msgs, n, **gen_kwargs)