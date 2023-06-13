from omegaconf import DictConfig
from typing import Dict
from tabulate import tabulate

import numpy as np
import time

from reLLMs.util import pytorch_util as ptu
from reLLMs.model.base import BaseModel
from reLLMs.task.base import BaseTask
from reLLMs.logger.base import BaseLogger


class LLMEvaluator:
    def __init__(
        self,
        cfg: DictConfig,
        model: BaseModel,
        task: BaseTask,
        logger: BaseLogger
    ):
        self.cfg = cfg
        self.model = model
        self.task = task
        self.logger = logger

        self.analyses = []

        # Init GPU
        ptu.init_gpu(
            use_gpu=not self.cfg.no_gpu,
            gpu_id=self.cfg.which_gpu
        )

        # Set random seed
        ptu.set_seed(self.cfg.seed)

        self.logger.log(f"Model: {self.model}", with_prefix=False)
        self.logger.log(f"Task: {self.task}", with_prefix=False)
        self.logger.log_variant('config.yaml', self.cfg)

    def evaluate(self, model, task, datum: Dict) -> Dict:
        question = task.format(datum)
        prompt = task.prompt + question
        result = self.model.complete(
            prompt=prompt,
            system_message=task.system_message,
        )
        analysis: Dict = task.analyse(datum, question, result)
        return analysis

    def run_evaluate_loop(self):
        self.start_time = time.time()
        for epoch, datum in enumerate(self.task.data['test']):
            analysis = self.evaluate(self.model, self.task, datum)
            self.perform_logging(analysis)
            self.analyses.append(analysis)
            summary = self.perform_summary(epoch)
        self.logger.log(summary, with_prefix=False)

    def perform_logging(self, analysis: Dict):

        self.logger.record_dict(analysis)
        self.logger.dump_tabular(with_prefix=False, with_timestamp=False)

    def perform_summary(self, epoch: int):

        total_tokens = np.sum([a["tokens"] for a in self.analyses])
        total_cost = np.sum([a["cost"] for a in self.analyses])
        total_correct = np.sum([a["evaluation"] for a in self.analyses])
        average_latency = np.mean([a["latency"] for a in self.analyses])
        aggregated_speed = total_tokens / np.sum([a["latency"] for a in self.analyses])
        accuracy = total_correct / len(self.analyses)

        summary = [
            ['Model', str(self.model)],
            ['Task', str(self.task)],
            ['Progress', f"{epoch+1}/{len(self.task.data['test'])}"],
            ['Accuracy', accuracy],
            ['Time', (time.time() - self.start_time) / 60],
            ['Total Tokens', total_tokens],
            ['Total Cost', total_cost],
            ['Average Latency', average_latency],
            ['Aggregated Speed', aggregated_speed],

        ]
        print("\033c" + tabulate(summary))
        return tabulate(summary)
