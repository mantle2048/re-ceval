from datasets.dataset_dict import DatasetDict
from typing import Dict

from reLLMs.util.result import Result


class BaseTask:
    """Base class for all datasets.
    Methods will raise NotImplementedError if they are not overwritten.
    """

    TASK_NAMES = []
    PROMPT_TYPE = ['vanilla', 'cot']

    def __init__(
        self,
        name: str,
        prompt_type: str = 'vanilla',
        few_shot: bool = False,
        **kwargs
    ):
        assert name in self.TASK_NAMES, \
            f"Invalid Task Name {name}"
        assert prompt_type in self.PROMPT_TYPE, \
            f"Invalid Prompt Type {prompt_type}"

        self.name = name
        self.prompt_type = prompt_type
        self.few_shot = few_shot

        self.data = self._load_dataset(name=name)
        self.prompt = self._create_prompt(prompt_type=prompt_type)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.name}')"

    def __str__(self):
        return f"{self.__class__.__name__}('{self.name}')"

    def format(self, datum: Dict) -> Dict:
        raise NotImplementedError

    def extract(self, answer: str) -> str:
        raise NotImplementedError

    def analyse(self, datum: Dict, result: Result) -> Dict:
        raise NotImplementedError

    def _load_dataset(self, name: str) -> DatasetDict:
        raise NotImplementedError

    def _create_prompt(self, prompt_type: str) -> str:
        raise NotImplementedError
