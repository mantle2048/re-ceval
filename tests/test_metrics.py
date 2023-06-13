import pytest
from reLLMs.metric import Metrics, name2metric


class TestMetrics:

    def test_metrics(self):

        metrics = Metrics(*name2metric.keys())
        results = metrics.compute(predictions=[1,2,3], references=[1,2,5])
        assert results['accuracy'] == 0.67
        results = metrics.compute(predictions=[1,2,3], references=[11,22,33])
        assert results['accuracy'] == 0.33
