# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from functools import wraps
import multiprocessing
import random
import re

from joblib import Parallel, delayed
import torch

from access.text import to_words
from access.utils.helpers import (open_files, yield_lines, yield_lines_in_parallel, get_temp_filepath, delete_files,
                                  get_temp_filepaths)


def apply_line_method_to_file(line_method, input_filepath):
    output_filepath = get_temp_filepath()
    with open(input_filepath, 'r') as input_file, open(output_filepath, 'w') as output_file:
        for line in input_file:
            transformed_line = line_method(line.rstrip('\n'))
            if transformed_line is not None:
                output_file.write(transformed_line + '\n')
    return output_filepath


def replace_lrb_rrb(text):
    text = re.sub(r'-lrb-', '(', text, flags=re.IGNORECASE)
    text = re.sub(r'-rrb-', ')', text, flags=re.IGNORECASE)
    text = re.sub(r'-lsb-', '[', text, flags=re.IGNORECASE)
    text = re.sub(r'-rsb-', ']', text, flags=re.IGNORECASE)
    text = re.sub(r'-lcb-', '{', text, flags=re.IGNORECASE)
    text = re.sub(r'-rcb-', '}', text, flags=re.IGNORECASE)
    return text


def replace_lrb_rrb_file(filepath):
    return apply_line_method_to_file(replace_lrb_rrb, filepath)


def to_lrb_rrb(text):
    # TODO: Very basic
    text = re.sub(r'((^| ))\( ', r'\1-lrb- ', text)
    text = re.sub(r' \)((^| ))', r' -rrb-\1', text)
    return text


def replace_back_quotes(text):
    return text.replace('`', "'")


def replace_double_quotes(text):
    return text.replace("''", '"')


def normalize_quotes(text):
    return replace_double_quotes(replace_back_quotes(text))


def to_lrb_rrb_file(input_filepath):
    return apply_line_method_to_file(to_lrb_rrb, input_filepath)


def lowercase_file(filepath):
    return apply_line_method_to_file(lambda line: line.lower(), filepath)


def concatenate_files(input_filepaths, output_filepath):
    with open(output_filepath, 'w') as output_f:
        for input_file in input_filepaths:
            with open(input_file, 'r') as input_f:
                for line in input_f:
                    output_f.write(line)


def split_file(input_filepath, output_filepaths, round_robin=False):
    if not round_robin:
        raise NotImplementedError('Splitting files is only implemented as round robin.')
    with open_files(output_filepaths, 'w') as files:
        # We write each line to a different file in a round robin fashion
        for i, line in enumerate(yield_lines(input_filepath)):
            files[i % len(output_filepaths)].write(line + '\n')


def merge_files(input_filepaths, output_filepath, round_robin=False):
    if not round_robin:
        return concatenate_files(input_filepaths, output_filepath)
    with open(output_filepath, 'w') as f:
        for lines in yield_lines_in_parallel(input_filepaths, strict=False):
            for line in lines:
                if line is None:
                    return
                f.write(line + '\n')


def get_real_n_jobs(n_jobs):
    n_cpus = multiprocessing.cpu_count()
    if n_jobs < 0:
        # Adopt same logic as joblib
        n_jobs = n_cpus + 1 + n_jobs
    if n_jobs > n_cpus:
        print('Setting n_jobs={n_jobs} > n_cpus={n_cpus}, setting n_jobs={n_cpus}')
        n_jobs = n_cpus
    assert 0 < n_jobs <= n_cpus
    return n_jobs


def get_parallel_file_pair_preprocessor(file_pair_preprocessor, n_jobs):
    if n_jobs == 1:
        return file_pair_preprocessor
    n_jobs = get_real_n_jobs(n_jobs)

    @wraps(file_pair_preprocessor)
    def parallel_file_pair_preprocessor(complex_filepath, simple_filepath, output_complex_filepath,
                                        output_simple_filepath):
        temp_complex_filepaths = get_temp_filepaths(n_jobs)
        temp_simple_filepaths = get_temp_filepaths(n_jobs)
        split_file(complex_filepath, temp_complex_filepaths, round_robin=True)
        split_file(simple_filepath, temp_simple_filepaths, round_robin=True)
        preprocessed_temp_complex_filepaths = get_temp_filepaths(n_jobs)
        preprocessed_temp_simple_filepaths = get_temp_filepaths(n_jobs)
        tasks = [
            delayed(file_pair_preprocessor)(*paths)
            for paths in zip(temp_complex_filepaths, temp_simple_filepaths, preprocessed_temp_complex_filepaths,
                             preprocessed_temp_simple_filepaths)
        ]
        Parallel(n_jobs=n_jobs)(tasks)
        merge_files(preprocessed_temp_complex_filepaths, output_complex_filepath, round_robin=True)
        merge_files(preprocessed_temp_simple_filepaths, output_simple_filepath, round_robin=True)
        delete_files(temp_complex_filepaths)
        delete_files(temp_simple_filepaths)
        delete_files(preprocessed_temp_complex_filepaths)
        delete_files(preprocessed_temp_simple_filepaths)

    return parallel_file_pair_preprocessor


def word_shuffle(words, max_swap=3):
    noise = torch.rand(len(words)).mul_(max_swap)
    permutation = torch.arange(len(words)).float().add_(noise).sort()[1]
    return [words[i] for i in permutation]


def word_dropout(words, dropout_prob=0.1):
    keep = torch.rand(len(words))
    dropped_out_words = [word for i, word in enumerate(words) if keep[i] > dropout_prob]
    if len(dropped_out_words) == 0:
        return [words[random.randint(0, len(words) - 1)]]
    return dropped_out_words


def word_blank(words, blank_prob=0.1):
    keep = torch.rand(len(words))
    return [word if keep[i] > blank_prob else '<BLANK>' for i, word in enumerate(words)]


def add_noise(sentence):
    words = to_words(sentence)
    words = word_shuffle(words, max_swap=3)
    words = word_dropout(words, dropout_prob=0.1)
    words = word_blank(words, blank_prob=0.1)
    return ' '.join(words)
