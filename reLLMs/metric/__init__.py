from typing import List, Any, Dict
from .accuracy import Accuracy

name2metric = {
    'accuarcy': Accuracy
}


class Metrics:

    def __init__(self, names: List):
        self.metrics = []
        for name in set(names).intersection(name2metric.keys()):
            self.metrics.append(name2metric[name]())

    def compute(
        self,
        predictions: List[Any],
        references: List[Any]
    ) -> Dict:
        results = {}
        for metric in self.metrics:
            ret = metric.compute(predictions, references)
            results.update(ret)
        return results
