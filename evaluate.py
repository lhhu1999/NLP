import numpy as np
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc, accuracy_score

def evaluate(total_loss, pred_labels, predictions, labels):
    LOSS = np.mean(total_loss)
    pred_labels = np.array(pred_labels)
    predictions = np.array(predictions)
    labels = np.array(labels)

    Acc = accuracy_score(labels, pred_labels)
    if sum(pred_labels) == 0:
        Pre = 0
        Recall = 0
        F1 = 0.0
    else:
        Pre = precision_score(labels, pred_labels)
        Recall = recall_score(labels, pred_labels)
        F1 = (2 * Pre * Recall) / (Pre + Recall)
    AUC = roc_auc_score(labels, predictions)
    tpr2, fpr2, s2 = precision_recall_curve(labels, predictions)
    AUPR = auc(fpr2, tpr2)

    Acc = float(format(Acc, '.4f'))
    Pre = float(format(Pre, '.4f'))
    Recall = float(format(Recall, '.4f'))
    AUC = float(format(AUC, '.4f'))
    AUPR = float(format(AUPR, '.4f'))
    F1 = float(format(F1, '.4f'))
    LOSS = float(format(LOSS, '.4f'))
    return [Acc, Pre, Recall, AUC, AUPR, F1, LOSS]


def save_result(item, res, epoch):
    print("%s>>> Acc: %f Pre: %f Recall: %f AUC: %f AUPR: %f F1: %f LOSS:%f" % (item, res[0], res[1], res[2], res[3], res[4], res[5], res[6]))

    dir_output = "./result/"
    os.makedirs(dir_output, exist_ok=True)
    file = "./result/{}.txt".format(item)
    with open(file, 'a') as f:
        res = [epoch] + res
        f.write('\t'.join(map(str, res)) + '\n')
        f.close()
