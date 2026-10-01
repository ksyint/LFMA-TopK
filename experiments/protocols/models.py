"""Pinned pretrained backbone metadata for reproducible experiments."""
BACKBONES = {
    'vit-base': {'family': 'vit', 'id': 'google/vit-base-patch16-224-in21k',
                 'revision': 'b4569560a39a0f1af58e3ddaf17facf20ab919b0', 'width': 768, 'layers': 12},
    'vit-large': {'family': 'vit', 'id': 'google/vit-large-patch16-224-in21k',
                  'revision': '6074eaf2211423e928c93b93ef773d5da618aa7e', 'width': 1024, 'layers': 24},
    'roberta-base': {'family': 'roberta', 'id': 'FacebookAI/roberta-base',
                     'revision': 'e2da8e2f811d1448a5b465c236feacd80ffbac7b', 'width': 768, 'layers': 12},
    'roberta-large': {'family': 'roberta', 'id': 'FacebookAI/roberta-large',
                      'revision': '722cf37b1afa9454edce342e7895e588b6ff1d59', 'width': 1024, 'layers': 24},
}
