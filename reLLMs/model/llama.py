import os
import torch

from transformers import LlamaTokenizer, LlamaForCausalLM
from pathlib import Path
from typing import Dict

import reLLMs.util.pytorch_util as ptu

from .base import BaseModel
from reLLMs.util.result import Result


class LLaMAModel(BaseModel):
    # cost is per million tokens
    MODEL_INFO = {
        "llama-7b-hf": {"prompt": 0.0, "completion": 0.0, "token_limit": 4096},
        "llama-13b-hf": {"prompt": 0.0, "completion": 0.0, "token_limit": 4096},
        "llama-65b-hf": {"prompt": 0.0, "completion": 0.0, "token_limit": 4096},
    }

    def __init__(
        self,
        ckpt_dir: str,
        name: str = None,
        temperature: float = 0,
        max_new_tokens: int = 20,
    ):
        if name is None:
            name = list(self.MODEL_INFO.keys())[0]
        assert name in self.MODEL_INFO.keys(), f'Invalid model name {name}'

        self.ckpt_dir = os.getenv(ckpt_dir)
        self.name = name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        model_path = Path(self.ckpt_dir) / self.name
        self.tokenizer, self.model = self._load_model(model_path)

    def count_tokens(self, content: str) -> int:
        result = self.tokenizer(content)
        return len(result.input_ids)

    def _prepapre_model_inputs(
        self,
        prompt: str,
        system_message: str = '',
        **kwargs,
    ) -> Dict:

        prompt = system_message + prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        model_inputs = {
            "input_ids": inputs.input_ids.to(ptu.device),
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            **kwargs,
        }
        return model_inputs

    def complete(
        self,
        prompt: str,
        system_message: str = '',
        **kwargs,
    ) -> Result:
        model_inputs = self._prepapre_model_inputs(
            prompt=prompt,
            system_message=system_message
        )
        with self.track_latency():
            generate_ids = self.model.generate(**model_inputs)

        completion = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        meta = {
            "latency": self.latency,
            "tokens_prompt": model_inputs['input_ids'].shape[0],
        }
        return Result(
            text=completion,
            model_inputs=model_inputs,
            model=self,
            meta=meta
        )

    def _load_model(self, path: str) -> (LlamaTokenizer, LlamaForCausalLM):
        tokenizer = LlamaTokenizer.from_pretrained(
            path,
            use_fast=False,
            padding_side="left",
        )
        model = LlamaForCausalLM.from_pretrained(
            path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        return tokenizer, model
