from unittest import TestCase

import numpy as np

from clase_1.query_mean_precision_K import QueryMeanPrecisionK


class QueryMeanPrecisionKTestCase(TestCase):

    def test_query_mean_precision(self):
        K = 3
        q_id = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4])
        predicted_rank = np.array([0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3])
        truth_relevance = np.array([True, False, True, False, True, True, True, False, False, False, False, False, True, False, False, True])
        expected = np.float64(0.5833333333333333)
        result = QueryMeanPrecisionK().__call__(predicted_rank, truth_relevance, q_id, K)
        np.testing.assert_equal(expected, result)
