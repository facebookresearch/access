# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from access.utils.helpers import harmonic_mean


# Tranforms take a value and cast it to a score between 0 and 1, the higher the better
def bleu_transform(bleu):
    min_bleu = 0
    max_bleu = 100
    bleu = max(bleu, min_bleu)
    bleu = min(bleu, max_bleu)
    return (bleu - min_bleu) / (max_bleu - min_bleu)


def sari_transform(sari):
    min_sari = 0
    max_sari = 60
    sari = max(sari, min_sari)
    sari = min(sari, max_sari)
    return (sari - min_sari) / (max_sari - min_sari)


def fkgl_transform(fkgl):
    min_fkgl = 0
    max_fkgl = 20
    fkgl = max(fkgl, min_fkgl)
    fkgl = min(fkgl, max_fkgl)
    return 1 - (fkgl - min_fkgl) / (max_fkgl - min_fkgl)


def combine_metrics(bleu, sari, fkgl, coefs):
    # Combine into a score between 0 and 1, LOWER the better
    assert len(coefs) == 3
    return 1 - harmonic_mean([bleu_transform(bleu), sari_transform(sari), fkgl_transform(fkgl)], coefs)
