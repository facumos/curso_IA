class Accuracy(object):

    def __call__(self, prediction, truth):
        """
        Definition:
        Is the sum of true positives (TP) and true negatives (TN)
        divided by the sum of all the parameters (TP, TN, FP, FN)
        Example:
        prediction = np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1])
        truth = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 1])
        tp = 2
        tn = 3
        fp = 2
        fn = 3
        tp + tn + fp + fn = 10
        accuracy = 5/10 = 0.5
        """

        true_pos_mask = (prediction == 1) & (truth == 1)
        true_pos = true_pos_mask.sum()
        true_neg_mask = (prediction == 0) & (truth == 0)
        true_neg = true_neg_mask.sum()
        false_pos_mask = (prediction == 1) & (truth == 0)
        false_pos = false_pos_mask.sum()
        false_neg_mask = (prediction == 0) & (truth == 1)
        false_neg = false_neg_mask.sum()
        return (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
