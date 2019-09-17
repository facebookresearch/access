# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from functools import wraps
from pathlib import Path
import shutil
import tempfile

from imohash import hashfile

from access.fairseq.base import fairseq_generate
from access.preprocessors import ComposedPreprocessor, load_preprocessors
from access.utils.helpers import count_lines


def memoize_simplifier(simplifier):
    memo = {}

    @wraps(simplifier)
    def wrapped(complex_filepath, pred_filepath):
        complex_filehash = hashfile(complex_filepath, hexdigest=True)
        previous_pred_filepath = memo.get(complex_filehash)
        if previous_pred_filepath is not None and Path(previous_pred_filepath).exists():
            assert count_lines(complex_filepath) == count_lines(previous_pred_filepath)
            # Reuse previous prediction
            shutil.copyfile(previous_pred_filepath, pred_filepath)
        else:
            simplifier(complex_filepath, pred_filepath)
        # Save prediction
        memo[complex_filehash] = pred_filepath

    return wrapped


def get_fairseq_simplifier(exp_dir, reload_preprocessors=False, **kwargs):
    '''Method factory'''
    @memoize_simplifier
    def fairseq_simplifier(complex_filepath, output_pred_filepath):
        # Trailing spaces for markdown formatting
        print('simplifier_type="fairseq_simplifier"  ')
        print(f'exp_dir="{exp_dir}"  ')
        fairseq_generate(complex_filepath, output_pred_filepath, exp_dir, **kwargs)

    preprocessors = None
    if reload_preprocessors:
        preprocessors = load_preprocessors(exp_dir)
    if preprocessors is not None:
        fairseq_simplifier = get_preprocessed_simplifier(fairseq_simplifier, preprocessors)
    return fairseq_simplifier


def get_preprocessed_simplifier(simplifier, preprocessors):
    composed_preprocessor = ComposedPreprocessor(preprocessors)

    @memoize_simplifier
    @wraps(simplifier)
    def preprocessed_simplifier(complex_filepath, output_pred_filepath):
        print(f'preprocessors={preprocessors}')
        preprocessed_complex_filepath = tempfile.mkstemp()[1]
        composed_preprocessor.encode_file(complex_filepath, preprocessed_complex_filepath)
        preprocessed_output_pred_filepath = tempfile.mkstemp()[1]
        simplifier(preprocessed_complex_filepath, preprocessed_output_pred_filepath)
        composed_preprocessor.decode_file(preprocessed_output_pred_filepath,
                                          output_pred_filepath,
                                          encoder_filepath=complex_filepath)

    preprocessed_simplifier.__name__ = f'{preprocessed_simplifier.__name__}_{composed_preprocessor.get_suffix()}'
    return preprocessed_simplifier
