from typing import Dict
from omegaconf import OmegaConf

from reLLMs.model.base import BaseModel


class Result:
    def __init__(
        self,
        text: str,
        model_inputs: Dict,
        model: BaseModel,
        meta: Dict = {},
    ):
        self._meta = meta
        self.text = text
        self.model = model
        self.model_inputs = model_inputs

    def __repr__(self) -> str:
        return OmegaConf.to_yaml(OmegaConf.create(self.to_dict()))

    @property
    def tokens_completion(self) -> int:
        if tokens_completion := self._meta.get("tokens_completion"):
            return tokens_completion
        return self.model.count_tokens(self.text)

    @property
    def tokens_prompt(self) -> int:
        if tokens_prompt := self._meta.get("tokens_prompt"):
            return tokens_prompt
        else:
            return self.model.count_tokens(self.model_inputs["prompt"])

    @property
    def tokens(self) -> int:
        return self.tokens_completion + self.tokens_prompt

    @property
    def cost(self) -> float:
        if cost := self._meta.get("cost"):
            return cost
        else:
            return self.model.compute_cost(
                prompt_tokens=self.tokens_prompt,
                completion_tokens=self.tokens_completion
            )

    @property
    def latency(self) -> float:
        if latency := self._meta.get("latency"):
            return latency
        else:
            return self.model.latency

    @property
    def meta(self) -> Dict:
        return {
            "model": self.model.name,
            "tokens": self.tokens,
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "cost": self.cost,
            "latency": self.latency
        }

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "meta": self.meta,
            "model_inputs": self.model_inputs,
            "class": str(self.model),
        }
