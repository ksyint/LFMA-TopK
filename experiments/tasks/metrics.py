"""GLUE metrics and image accuracy from held-out model predictions."""
import numpy as np


def task_metrics(task, predictions, labels):
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    predictions, labels = np.asarray(predictions), np.asarray(labels)
    if task == 'stsb':
        if np.std(predictions) == 0 or np.std(labels) == 0 or len(labels) < 2:
            return {'pearson': 0.0, 'spearmanr': 0.0}
        return {'pearson': float(pearsonr(predictions, labels).statistic),
                'spearmanr': float(spearmanr(predictions, labels).statistic)}
    values = {'accuracy': float(accuracy_score(labels, predictions))}
    if task == 'cola':
        values['matthews_correlation'] = float(matthews_corrcoef(labels, predictions))
    if task == 'mrpc':
        values['f1'] = float(f1_score(labels, predictions, zero_division=0))
    return values
