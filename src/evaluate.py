import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)


def get_all_metrics(labels, probs, threshold=0.5):
    preds  = (probs > threshold).astype(int)
    report = classification_report(labels, preds,
                                   target_names=['Human', 'Bot'],
                                   output_dict=True)
    cm     = confusion_matrix(labels, preds)
    fpr, tpr, _ = roc_curve(labels, probs)
    prec, rec, _ = precision_recall_curve(labels, probs)
    return {
        'report':  report,
        'cm':      cm,
        'fpr':     fpr,   'tpr': tpr,   'roc_auc': auc(fpr, tpr),
        'prec':    prec,  'rec': rec,   'pr_auc':  auc(rec, prec)
    }