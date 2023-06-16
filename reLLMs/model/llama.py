import os
import torch

from transformers import LlamaTokenizer, LlamaForCausalLM
from pathlib import Path
from typing import Dict

import reLLMs.util.pytorch_util as ptu

from .base import BaseModel
from reLLMs.util.result import Result

ptu.device = torch.device("cuda:0")


class LLaMAModel(BaseModel):
    # cost is per million tokens
    MODEL_INFO = {
        "llama-7b": {"prompt": 0.0, "completion": 0.0, "token_limit": 4096},
    }

    def __init__(
        self,
        ckpt_dir: str,
        name: str = None,
        temperature: float = 0,
        max_tokens: int = 20,
    ):
        if name is None:
            name = list(self.MODEL_INFO.keys())[0]
        assert name in self.MODEL_INFO.keys(), f'Invalid model name {name}'

        self.ckpt_dir = os.getenv(ckpt_dir)
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens

        model_path = Path(self.ckpt_dir) / self.name
        self.tokenizer, self.model = self._load_model(model_path)

    def count_tokens(self, content: str) -> int:
        result = self.tokenizer(content)
        return len(result.input_ids[0])

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
            "max_length": self.max_tokens,
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
        }
        print(completion)
        return Result(
            text=completion,
            model_inputs=model_inputs,
            model=self,
            meta=meta
        )

    def _load_model(self, path: str) -> (LlamaTokenizer, LlamaForCausalLM):
        max_memory_mapping = {4: '10GB', 5: '10GB', 6: '10GB', 7: '10GB'}
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(
            path,
            device_map="auto",
            # load_in_8bit=True,
            max_memory=max_memory_mapping
        )
        return tokenizer, model
