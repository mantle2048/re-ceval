from typing import Dict, List
from datasets.dataset_dict import DatasetDict
from datasets import load_dataset

from .base import BaseTask
from reLLMs.util.result import Result
from reLLMs.util.text_util import find_words_last_idx

import re
import warnings


class CEvalTask(BaseTask):
    TASK_NAMES = [
        "computer_network",
        "operating_system",
        "computer_architecture",
        "college_programming",
        "college_physics",
        "college_chemistry",
        "advanced_mathematics",
        "probability_and_statistics",
        "discrete_mathematics",
        "electrical_engineer",
        "metrology_engineer",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_chemistry",
        "high_school_biology",
        "middle_school_mathematics",
        "middle_school_biology",
        "middle_school_physics",
        "middle_school_chemistry",
        "veterinary_medicine",
        "college_economics",
        "business_administration",
        "marxism",
        "mao_zedong_thought",
        "education_science",
        "teacher_qualification",
        "high_school_politics",
        "high_school_geography",
        "middle_school_politics",
        "middle_school_geography",
        "modern_chinese_history",
        "ideological_and_moral_cultivation",
        "logic",
        "law",
        "chinese_language_and_literature",
        "art_studies",
        "professional_tour_guide",
        "legal_professional",
        "high_school_chinese",
        "high_school_history",
        "middle_school_history",
        "civil_servant",
        "sports_science",
        "plant_protection",
        "basic_medicine",
        "clinical_medicine",
        "urban_and_rural_planner",
        "accountant",
        "fire_engineer",
        "environmental_impact_assessment_engineer",
        "tax_accountant",
        "physician"
    ]

    def __init__(
        self,
        name: str,
        prompt_type: str = 'vanilla',
        few_shot: bool = False,
    ):
        self.choices = ['A', 'B', 'C', 'D']
        self.system_message = \
            f"你是一个中文人工智能助手，以下是中国关于{name}考试的单项选择题，请选出其中的正确答案。"
        super().__init__(name, prompt_type, few_shot)

    def format(self, datum: Dict, include_ans=False) -> str:
        ret = datum['question']
        for choice in self.choices:
            ret += f"\n{choice}. " + datum[choice]
        ret += self._answer_prefix
        if include_ans:
            ret += self._answer_core.format(
                explanation=datum['explanation'],
                answer=datum['answer']
            ) + "\n\n"
        return ret

    def extract(self, ans: str) -> str:
        pattern = [
            r"^选([A-D])",
            r"^选项([A-D])",
            r"答案\s?选?项?\s?为?([A-D])",
            r"答案是\s?选?项?\s?([A-D])",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:?\s?选?项?\s?([A-D])",
            r"答案应该是:?\s?选?项?\s?([A-D])",
            r"正确的一项是\s?([A-D])",
            r"正确答案是:?\s?([A-D])",
            r"答案为:\s?选?项?\s?([A-D])",
            r"答案应为:\s?选?项?\s?([A-D])",
            r"答案:\s?选?项?\s?([A-D])",
            r"答案是：?\s?选?项?\s?([A-D])",
            r"答案应该是：?\s?选?项?\s?([A-D])",
            r"答案为：?\s?选?项?\s?([A-D])",
            r"答案应为：?\s?选?项?\s?([A-D])",
            r"答案：\s?选?项?\s?([A-D])",
        ]
        # if answer line only consists a choice
        ans_list = []
        for p in pattern:
            ans_list += re.findall(p, ans)
        if len(set(ans_list)) == 0 :
            letter_pattern = r"([A-D])\."
            conclusion_words = ['因此', '综上', '所以', '结论']
            idx = find_words_last_idx(ans, conclusion_words)
            import ipdb; ipdb.set_trace()
            if idx != -1:
                patterns = re.findall(letter_pattern, ans[idx:])
            else:
                patterns = re.findall(letter_pattern, ans)
            ans_list += patterns
        if len(set(ans_list)) > 1:
            warnings.warn("Model outputs multiple choices, only return the last one.")
        try:
            return ans_list[-1]
        except IndexError:
            warnings.warn("Model doesn't output any choices, return None.")
            return None

    def analyse(self, datum: Dict, question: str, result: Result) -> Dict:
        analysis = {}
        ans_model, ans_target = self.extract(result.text), datum['answer']
        evaluation = 1 if (ans_model == ans_target) and (ans_model is not None) else 0
        analysis = {
            'id': datum['id'],
            'evaluation': evaluation,
            'question': question.replace('\n', ' '),
            'answer_target': ans_target,
            'answer_model': ans_model,
            'explanation': datum['explanation'].replace('\n', ' '),
            'completion': result.text.replace('\n', ' '),
            **result.meta,
        }
        return analysis

    def _load_dataset(self, name: str) -> DatasetDict:
        return load_dataset("ceval/ceval-exam", name=name)

    def _create_prompt(self, few_shot: bool) -> str:
        prompt = self.system_message + "\n\n"
        if few_shot:
            for datum in self.data['dev']:
                prompt += self.format(datum, include_ans=True)
        return prompt

    @property
    def _answer_prefix(self) -> str:
        if self.prompt_type == 'vanilla':
            return "\n答案："
        elif self.prompt_type == 'cot':
            return "\n答案：让我们一步一步思考，\n"
        else:
            return ""

    @property
    def _answer_core(self) -> str:
        if self.prompt_type == 'vanilla':
            return "{answer}"
        elif self.prompt_type == 'cot':
            return "{explanation}" + "\n所以答案是{answer}"
        else:
            return ""
