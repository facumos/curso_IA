import numpy as np

""""
Este archivo lo cree antes de ver las soluciones para este ejercicio y darme cuenta que 
hay que hacerlo con clases.
"""""


def obtain_var(truth, prediction):
    out = np.array([np.sum(np.logical_and(truth == 1, prediction == 1))])  # True Positive
    out = np.append(out, np.sum(np.logical_and(truth == 0, prediction == 0)))  # True Negative
    out = np.append(out, np.sum(np.logical_and(truth == 1, prediction == 0)))  # False Positive
    out = np.append(out, np.sum(np.logical_and(truth == 0, prediction == 1)))  # False Negative
    return out


def calc_precision(variables):
    out = variables[0]/(variables[0]+variables[2])  # TP/(TP+FP)
    return out


def calc_recall(variables):
    out = variables[0]/(variables[0]+variables[3])  # TP/(TP+FN)
    return out


def calc_accuracy(variables):
    out = (variables[0]+variables[1]) / (np.sum(variables))  # (TP+TN)/(TP+TN+FP+FN)
    return out

