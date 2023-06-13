import evaluate
import numpy as np

from typing import Dict, List


class Accuracy(evaluate.Metric):

    def __init__(self):
        self.correct = 0
        self.total = 0

    def compute(self, predictions: List, references: List) -> Dict:
        score = np.array(predictions) == np.array(references)
        self.correct += sum(score)
        self.total += len(score)
        accuracy = round(self.correct / self.total, 2)
        return {
            "Accuracy": accuracy
        }
