import os
import time
from contextlib import contextmanager


class BaseModel:
    """Base class for all foundation models.
    Methods will raise NotImplementedError if they are not overwritten.
    """

    MODEL_INFO = {}

    def __init__(self, api_key: str = None, name: str = None, **kwargs):
        self.name = name
        self.api_key = os.getenv(api_key)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.name}')"

    def __str__(self):
        return f"{self.__class__.__name__}('{self.name}')"

    @contextmanager
    def track_latency(self):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.latency = round(time.perf_counter() - start, 2)

    def compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        cost_per_token = self.MODEL_INFO[self.name]
        cost = (
            (prompt_tokens * cost_per_token["prompt"])
            + (completion_tokens * cost_per_token["completion"])
        ) / 1_000_000
        cost = round(cost, 5)
        return cost

    def count_tokens(self):
        raise NotImplementedError(
            f"Count tokens is currently not supported with {self.__name__}"
        )

    def complete(self):
        raise NotImplementedError

    async def acomplete(self):
        raise NotImplementedError(
            f"Async complete is not yet supported with {self.__name__}"
        )
