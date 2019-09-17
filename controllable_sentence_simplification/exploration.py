from functools import lru_cache

import pandas as pd

from ts.text import to_words
from ts.utils.helpers import yield_lines_in_parallel


def get_lcs(seq1, seq2):
    '''Returns the longest common subsequence using memoization (only in local scope)'''
    @lru_cache(maxsize=None)
    def recursive_lcs(seq1, seq2):
        if len(seq1) == 0 or len(seq2) == 0:
            return []
        if seq1[-1] == seq2[-1]:
            return recursive_lcs(seq1[:-1], seq2[:-1]) + [seq1[-1]]
        else:
            return max(recursive_lcs(seq1[:-1], seq2), recursive_lcs(seq1, seq2[:-1]), key=lambda seq: len(seq))

    try:
        return recursive_lcs(tuple(seq1), tuple(seq2))
    except RecursionError as e:
        print(e)
        # TODO: Handle this case
        return []


def compare_sentences(complex_sent, simple_sent, make_bold=lambda word: f'**{word}**'):
    # Returns the two sentences with different words in bold (markdown)
    def format_words(words, mutual_words):
        '''Makes all words bold except the mutual ones'''
        words_generator = iter(words)
        formatted_words = []
        for mutual_word in mutual_words:
            word = next(words_generator)
            while word != mutual_word:
                formatted_words.append(make_bold(word))
                word = next(words_generator)
            formatted_words.append(word)
        # Add remaining words
        formatted_words.extend([make_bold(word) for word in words_generator])
        return ' '.join(formatted_words)

    complex_words = to_words(complex_sent)
    simple_words = to_words(simple_sent)
    mutual_words = get_lcs(complex_words, simple_words)
    return format_words(complex_words, mutual_words), format_words(simple_words, mutual_words)


def write_comparison_file(complex_filepath, simple_filepath, comparison_filepath, sort_key=None,
                          n_samples=float('inf')):
    with open(comparison_filepath, 'w') as f:
        separator = '\n' + '-' * 80
        pair_generator = yield_lines_in_parallel([complex_filepath, simple_filepath])
        if sort_key is not None:
            pair_generator = sorted(pair_generator, key=lambda args: sort_key(*args))
        for i, (complex_line, simple_line) in enumerate(pair_generator):
            if i >= n_samples:
                break
            # This is markdown formatted
            complex_line, simple_line = compare_sentences(complex_line, simple_line)
            f.write(f'{separator}  \n{complex_line}  \n{simple_line}  \n')


def df_append_row(df, row, row_name=None):
    if row_name is None:
        return df.append(pd.Series(row), ignore_index=True)
    else:
        return df.append(pd.Series(row, name=row_name))
