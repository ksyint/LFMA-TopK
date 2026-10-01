"""GLUE fields, primary metrics, and the pinned data release."""
GLUE_TASKS = {
    'sst2': ('sentence', None, 'accuracy'), 'mrpc': ('sentence1', 'sentence2', 'accuracy'),
    'qnli': ('question', 'sentence', 'accuracy'), 'rte': ('sentence1', 'sentence2', 'accuracy'),
    'cola': ('sentence', None, 'matthews_correlation'),
    'stsb': ('sentence1', 'sentence2', 'pearson'),
}
GLUE_REVISION = 'bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c'
