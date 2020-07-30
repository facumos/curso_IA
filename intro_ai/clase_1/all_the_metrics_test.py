from unittest import TestCase

import numpy as np

from clase_1.all_the_metrics import BaseMetrics, Prediction, Recall, Accuracy, QueryMeanPrecision, QueryMeanPrecisionK

class AllTheMetrics(TestCase):

    def test_all_the_metrics(self):
        k = 1
        truth = np.array([1, 1, 0, 1, 1, 1, 0, 0, 0, 1])
        prediction = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0])
        q_id = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4])
        predicted_rank = np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3])
        truth_relevance = np.array([True, False, True, False, True, True, True, False, False, False, False, False, True, False, False, True])
        BaseMetrics(truth, prediction, predicted_rank, truth_relevance, q_id, k)

        expected = np.float64(0.5)
        result = Prediction.__call__(true_pos, false_pos)
        np.testing.assert_equal(expected, result)


