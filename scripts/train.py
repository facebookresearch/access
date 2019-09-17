# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from access.fairseq.main import fairseq_train_and_evaluate
from access.resources.prepare import prepare_wikilarge, prepare_turkcorpus


if __name__ == '__main__':
    print('Training a model from scratch')
    prepare_wikilarge()
    prepare_turkcorpus()
    kwargs = {
        'arch': 'transformer',
        'warmup_updates': 4000,
        'parametrization_budget': 256,
        'beam': 8,
        'dataset': 'wikilarge',
        'dropout': 0.2,
        'fp16': False,
        'label_smoothing': 0.54,
        'lr': 0.00011,
        'lr_scheduler': 'fixed',
        'max_epoch': 100,
        'max_tokens': 5000,
        'metrics_coefs': [0, 1, 0],
        'optimizer': 'adam',
        'preprocessors_kwargs': {
            'LengthRatioPreprocessor': {
                'target_ratio': 0.8  # Default initial value
            },
            'LevenshteinPreprocessor': {
                'target_ratio': 0.8  # Default initial value
            },
            'WordRankRatioPreprocessor': {
                'target_ratio': 0.8  # Default initial value
            },
            'DependencyTreeDepthRatioPreprocessor': {
                'target_ratio': 0.8  # Default initial value
            },
            'SentencePiecePreprocessor': {
                'vocab_size': 10000
            }
        }
    }
    fairseq_train_and_evaluate(**kwargs)
