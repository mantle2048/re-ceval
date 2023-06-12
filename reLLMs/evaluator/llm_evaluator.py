from tqdm import tqdm
from omegaconf import DictConfig
from typing import Dict

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
        epoch = 0
        for datum in tqdm(self.task.data['test']):
            analysis = self.evaluate(self.model, self.task, datum)
            epoch += 1
            self.perform_logging(analysis)
            if epoch == 2: break

    def perform_logging(self, analysis: Dict):

        self.logger.record_dict(analysis)
        self.logger.dump_tabular(with_prefix=False, with_timestamp=False)
