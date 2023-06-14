import os
import tiktoken
import warnings
import openai

from typing import Optional, List, Dict
from aiohttp import ClientSession
from tenacity import retry, wait_chain, wait_fixed

from .base import BaseModel
from reLLMs.util.result import Result


class OpenAIModel(BaseModel):
    # cost is per million tokens
    MODEL_INFO = {
        "gpt-3.5-turbo": {"prompt": 1.5, "completion": 2.0, "token_limit": 4096},
        "gpt-3.5-turbo-16k": {"prompt": 3.0, "completion": 4.0, "token_limit": 16_384},
        "gpt-4": {"prompt": 30.0, "completion": 60.0, "token_limit": 8192},
        "text-davinci-003": {"prompt": 20.0, "completion": 20.0, "token_limit": 4097},
    }

    def __init__(
        self,
        api_key: str,
        name: str = None,
        temperature: float = 0,
        max_tokens: int = 256,
    ):
        openai.api_key = os.getenv(api_key)

        if name is None:
            name = list(self.MODEL_INFO.keys())[0]
        assert name in self.MODEL_INFO.keys(), f'Invalid model name {name}'

        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = openai.ChatCompletion \
            if self.is_chat_model else openai.Completion

    @property
    def is_chat_model(self) -> bool:
        return self.name.startswith("gpt")

    def count_tokens(self, content: str) -> int:
        enc = tiktoken.encoding_for_model(self.name)
        return len(enc.encode(content))

    def _prepapre_model_inputs(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: str = None,
        **kwargs,
    ) -> Dict:
        if self.is_chat_model:
            messages = [{"role": "user", "content": prompt}]

            if history:
                messages = history + messages

            if system_message:
                messages = [{"role": "system", "content": system_message}] + messages

            model_inputs = {
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **kwargs,
            }
        else:
            if history:
                warnings.warn(
                    f"history argument is not supported for {self.name} model"
                )

            if system_message:
                warnings.warn(
                    f"system_message argument is not supported for {self.name} model"
                )

            model_inputs = {
                "prompt": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **kwargs,
            }
        return model_inputs

    @retry(wait=wait_chain(*[wait_fixed(3) for i in range(3)] +
                        [wait_fixed(5) for i in range(2)] +
                        [wait_fixed(10)]))
    def complete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: str = None,
        **kwargs,
    ) -> Result:
        """
        Args:
            history: messages in OpenAI format, each dict must include role and content key.
            system_message: system messages in OpenAI format, must have role and content key.
              It can has name key to include few-shots examples.
        """

        model_inputs = self._prepapre_model_inputs(
            prompt=prompt,
            history=history,
            system_message=system_message,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs,
        )

        with self.track_latency():
            response = self.client.create(model=self.name, **model_inputs)

        if self.is_chat_model:
            completion = response.choices[0].message.content.strip()
        else:
            completion = response.choices[0].text.strip()

        usage = response.usage

        meta = {
            "tokens_prompt": usage["prompt_tokens"],
            "tokens_completion": usage["completion_tokens"],
            "latency": self.latency,
        }
        return Result(
            text=completion,
            model_inputs=model_inputs,
            model=self,
            meta=meta
        )

    async def acomplete(
        self,
        prompt: str,
        history: Optional[List[dict]] = None,
        system_message: Optional[List[dict]] = None,
        **kwargs,
    ) -> Result:
        """
        Args:
            history: messages in OpenAI format, each dict must include role and content key.
            system_message: system messages in OpenAI format, must have role and content key.
              It can has name key to include few-shots examples.
        """

        async with ClientSession(trust_env=True) as session:
            openai.aiosession.set(session)

            model_inputs = self._prepapre_model_inputs(
                prompt=prompt,
                history=history,
                system_message=system_message,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs,
            )

            with self.track_latency():
                response = await self.client.acreate(model=self.name, **model_inputs)

        if self.is_chat_model:
            completion = response.choices[0].message.content.strip()
        else:
            completion = response.choices[0].text.strip()

        usage = response.usage

        meta = {
            "tokens_prompt": usage["prompt_tokens"],
            "tokens_completion": usage["completion_tokens"],
            "latency": self.latency,
        }
        return Result(
            text=completion,
            model_inputs=model_inputs,
            model=self,
            meta=meta,
        )
