import numpy as np


class BaseMetrics(object):

    # instance = None
    # def __new__(cls, name):
    #     if Metrics.instance is None:
    #         print("__new__ object created")
    #         Metrics.instance = super(Metrics,cls).__new__(cls)
    #         return Metrics.instance
    #     else:
    #         return Metrics.instance
    def __init__(self, prediction, truth, predicted_rank, truth_relevance, query_ids, K):
        true_pos_mask = (prediction == 1) & (truth == 1)
        self.true_pos = true_pos_mask.sum()
        true_neg_mask = (prediction == 0) & (truth == 0)
        self.true_neg = true_neg_mask.sum()
        false_pos_mask = (prediction == 1) & (truth == 0)
        self.false_pos = false_pos_mask.sum()
        false_neg_mask = (prediction == 0) & (truth == 1)
        self.false_neg = false_neg_mask.sum()
        # get count of queries with at least one true relevant document
        true_relevance_mask = (truth_relevance == 1)
        filtered_query_id = query_ids[true_relevance_mask]
        filtered_true_relevance_count = np.bincount(filtered_query_id)  # en las queries con relevance, cuento
        # complete the count of queries with zeros in queries without true relevant documents
        unique_query_ids = np.unique(query_ids)
        non_zero_count_idxs = np.where(filtered_true_relevance_count > 0)
        true_relevance_count = np.zeros(unique_query_ids.max() + 1)
        true_relevance_count[non_zero_count_idxs] = filtered_true_relevance_count[non_zero_count_idxs]  # No entiendo el
        # porque de esta linea
        # get the count only for existing queries
        self.true_relevance_count_by_query = true_relevance_count[unique_query_ids]
        # get the count of fetched documents
        self.fetched_documents_count = np.bincount(query_ids)[unique_query_ids]
        self.K = K


class Prediction(BaseMetrics):
    def __init__(self):
        BaseMetrics.__init__(self, prediction, truth, predicted_rank, truth_relevance, query_ids, K)

    def __call__(self, true_pos, false_pos):
        return true_pos / (true_pos + false_pos)


class Recall(BaseMetrics):
    def __call__(self, true_pos, false_neg):
        return true_pos / (true_pos + false_neg)


class Accuracy(BaseMetrics):
    def __call__(self, true_pos, true_neg, false_pos, false_neg):
        return (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)


class QueryMeanPrecision(BaseMetrics):
    def __call__(self, true_relevance_count_by_query, fetched_documents_count):
        precision_by_query = true_relevance_count_by_query / fetched_documents_count
        return np.mean(precision_by_query)


class QueryMeanPrecisionK(BaseMetrics):
    def __call__(self, true_relevance_count_by_query, K):
        precision_by_query = true_relevance_count_by_query / K
        return np.mean(precision_by_query)
