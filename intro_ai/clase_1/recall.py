class Recall(object):

    def __call__(self, prediction, truth):
        """
        Definition:
        Is the number of true positives (TP) divided by the sum of true positives (TP) and false negatives (FN)
        Example:
        prediction = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1])
        truth = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 1])
        tp = 2
        fn = 3
        tp + fn = 5
        recall = 0.2
        """

        true_pos_mask = (prediction == 1) & (truth == 1)
        true_pos = true_pos_mask.sum()
        false_neg_mask = (prediction == 0) & (truth == 1)
        false_neg = false_neg_mask.sum()
        return true_pos / (true_pos + false_neg)
