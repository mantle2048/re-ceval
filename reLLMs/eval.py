import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from reLLMs.evaluator.llm_evaluator import LLMEvaluator


@hydra.main(version_base=None, config_path='../cfgs', config_name='eval')
def main(cfg: DictConfig):
    logger = instantiate(cfg.logger)
    task = instantiate(cfg.task)
    model = instantiate(cfg.model)
    evaluator = LLMEvaluator(cfg, model, task, logger)
    evaluator.run_evaluate_loop()


if __name__ == '__main__':
    main()
