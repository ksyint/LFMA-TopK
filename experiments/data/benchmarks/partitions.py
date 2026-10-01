"""Fixed stratified image partitions across optimization seeds."""
import numpy as np

def stratified_indices(labels, fractions, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    groups = [[] for _ in fractions]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        cuts = np.cumsum([round(len(indices) * fraction) for fraction in fractions[:-1]])
        for group, values in zip(groups, np.split(indices, cuts)):
            group.extend(values.tolist())
    for group in groups:
        rng.shuffle(group)
        if not group:
            raise ValueError('Dataset split is empty. Adjust its fractions')
    return groups
