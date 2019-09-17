from pathlib import Path
import shutil
import tempfile
import time

from imohash import hashfile
import pandas as pd
from tabulate import tabulate
from easse.report import get_all_scores
from easse.cli import easse_report
from easse.utils.resources import get_turk_refs_sents, get_turk_orig_sents

from ts.exploration import write_comparison_file, df_append_row
from ts.feature_extraction import get_levenshtein_distance, count_sentence_splits, compression_ratio
from ts.preprocess import lowercase_file, to_lrb_rrb_file
from ts.preprocessors import markdown_escape_special_tokens
from ts.resources.paths import get_data_filepath, VARIOUS_DIR
from ts.utils.helpers import (read_file, get_line_lengths, yield_lines, count_lines, yield_lines_in_parallel, mute,
                              read_lines, get_temp_filepath)
'''A simplifier is a method with signature: simplifier(complex_filepath, output_pred_filepath)'''


def get_simplification_scores(source_filepath, pred_filepath, ref_filepaths):
    return get_all_scores(
        read_lines(source_filepath),
        read_lines(pred_filepath),
        [read_lines(ref_filepath) for ref_filepath in ref_filepaths],
    )


def get_prediction_on_turkcorpus(simplifier, phase):
    # orig_sents = get_turk_orig_sents(phase=phase)
    # HACK: We use the truecased data as input (the tokenization is also slightly different)
    # source_filepath = get_temp_filepath()
    # write_lines(orig_sents, source_filepath)
    # source_filepath = apply_line_method_to_file(replace_lrb_rrb, source_filepath)
    source_filepath = get_data_filepath('turkcorpus', phase, 'complex')
    pred_filepath = get_temp_filepath()
    with mute():
        simplifier(source_filepath, pred_filepath)
    return pred_filepath


def evaluate_simplifier_on_turkcorpus(simplifier, phase):
    pred_filepath = get_prediction_on_turkcorpus(simplifier, phase)
    pred_filepath = lowercase_file(pred_filepath)
    pred_filepath = to_lrb_rrb_file(pred_filepath)
    refs_sents = get_turk_refs_sents(phase=phase)
    orig_sents = get_turk_orig_sents(phase=phase)
    return get_all_scores(orig_sents, read_lines(pred_filepath), refs_sents)


def get_easse_report_on_turkcorpus(simplifier):
    pred_filepath = get_prediction_on_turkcorpus(simplifier, 'valid')
    pred_filepath = lowercase_file(pred_filepath)
    pred_filepath = to_lrb_rrb_file(pred_filepath)
    report_path = get_temp_filepath()
    easse_report('turk_valid', input_path=pred_filepath, report_path=report_path)
    return report_path


def get_sanity_check_text(simplifier, complex_filepath=VARIOUS_DIR / 'ts_examples.complex'):
    '''Displays input sentence, intermediary sentences and pred sentences side by side'''
    temp_dir = Path(tempfile.mkdtemp())

    def mocked_mkstemp():
        '''Mock tempfile.mkstemp() by creating the file in a specific folder with a timestamp in order to track them'''
        path = temp_dir / str(time.time())
        path.touch()
        return 0, path

    original_mkstemp = tempfile.mkstemp
    tempfile.mkstemp = mocked_mkstemp
    timestamped_complex_filepath = tempfile.mkstemp()[1]
    shutil.copyfile(complex_filepath, timestamped_complex_filepath)
    with open(timestamped_complex_filepath, 'a') as f:
        f.write(f'We add this line with a timestamp {time.time()} to change the file hash to prevent memoization .\n')
    pred_filepath = tempfile.mkstemp()[1]
    simplifier(timestamped_complex_filepath, pred_filepath)
    tempfile.mkstemp = original_mkstemp
    # Get temporary files that were created
    created_paths = sorted(temp_dir.glob('*'), key=lambda path: path.stat().st_mtime)
    # Remove duplicate files and empty files
    hashes = []
    paths = []
    n_complex_lines = count_lines(timestamped_complex_filepath)
    for path in [timestamped_complex_filepath] + created_paths + [pred_filepath]:
        if count_lines(path) != n_complex_lines:
            continue
        file_hash = hashfile(path)
        if file_hash in hashes:
            continue
        paths.append(path)
        hashes.append(file_hash)
    output_lines = []
    for lines in yield_lines_in_parallel(paths):
        output_lines += ['\n' + '-' * 80] + lines
    sep = '\n' + '-' * 10 + '\n'
    return f'{sep}## Sanity check  \n' + markdown_escape_special_tokens('  \n'.join(output_lines))


def evaluate_simplifier_qualitatively(simplifier):
    # Cherry picked complex sentences
    complex_filepath = VARIOUS_DIR / 'ts_examples.complex'
    _, pred_filepath = tempfile.mkstemp()
    _, comparison_filepath = tempfile.mkstemp()
    simplifier(complex_filepath, pred_filepath)
    write_comparison_file(complex_filepath, pred_filepath, comparison_filepath)
    output_text = '## Qualitative evaluation  \n'
    sep = '\n' + '-' * 10 + '\n'
    output_text += f'{sep}## Cherry picked complex sentences  \n'
    output_text += read_file(comparison_filepath)
    # Wikilarge predictions sorted with given sort_key
    complex_filepath = get_data_filepath('wikilarge', 'test', 'complex')
    _, pred_filepath = tempfile.mkstemp()
    simplifier(complex_filepath, pred_filepath)
    text_key = [
        ('Random Wikilarge predictions', lambda c, s: 0),
        ('Wikilarge predictions with the most sentence splits', lambda c, s: -count_sentence_splits(c, s)),
        ('Wikilarge predictions with the lowest compression ratio', lambda c, s: compression_ratio(c, s)),
        ('Wikilarge predictions with the highest Levenshtein distances', lambda c, s: -get_levenshtein_distance(c, s)),
    ]
    for text, sort_key in text_key:
        _, comparison_filepath = tempfile.mkstemp()
        write_comparison_file(complex_filepath, pred_filepath, comparison_filepath, sort_key=sort_key, n_samples=10)
        output_text += f'{sep}## {text}  \n'
        output_text += read_file(comparison_filepath)
    return markdown_escape_special_tokens(output_text)


def evaluate_simplifier_by_sentence_length(simplifier, n_bins=5):
    def get_intervals_from_limits(limits):
        return list(zip(limits[:-1], limits[1:]))

    def get_equally_populated_intervals(filepath, n_bins):
        line_lengths = sorted(get_line_lengths(filepath))
        n_samples_per_bin = int(len(line_lengths) / n_bins)
        limits = [line_lengths[i * n_samples_per_bin] for i in range(n_bins)] + [line_lengths[-1] + 1]
        return get_intervals_from_limits(limits)

    def split_lines_by_lengths(filepath, intervals):
        bins = [[] for _ in range(len(intervals))]
        for line_idx, line in enumerate(yield_lines(filepath)):
            line_length = len(line)
            for interval_idx, (interval_start, interval_end) in enumerate(intervals):
                if interval_start <= line_length and line_length < interval_end:
                    bins[interval_idx].append(line_idx)
                    break
        assert sum([len(b) for b in bins]) == count_lines(filepath)
        return bins

    def select_lines(input_filepath, output_filepath, line_indexes):
        line_indexes = set(line_indexes)
        with open(output_filepath, 'w') as f:
            for line_idx, line in enumerate(yield_lines(input_filepath)):
                if line_idx in line_indexes:
                    f.write(line + '\n')

    def split_file_by_bins(input_filepath, bins):
        splitted_filepaths = [tempfile.mkstemp()[1] for _ in range(len(bins))]
        for splitted_filepath, line_indexes in zip(splitted_filepaths, bins):
            select_lines(input_filepath, splitted_filepath, line_indexes)
        return splitted_filepaths

    # Run predicition
    complex_filepath = get_data_filepath('wikilarge', 'test', 'complex')
    ref_filepath = get_data_filepath('wikilarge', 'test', 'simple')
    _, pred_filepath = tempfile.mkstemp()
    simplifier(complex_filepath, pred_filepath)
    # Get line length bins
    intervals = get_equally_populated_intervals(complex_filepath, n_bins)
    bins = split_lines_by_lengths(complex_filepath, intervals)
    # Split files by bins
    splitted_complex_filepaths = split_file_by_bins(complex_filepath, bins)
    splitted_ref_filepaths = split_file_by_bins(ref_filepath, bins)
    splitted_pred_filepaths = split_file_by_bins(pred_filepath, bins)
    df_bins = pd.DataFrame()
    # Get scores for each bin
    for i in range(len(intervals)):
        interval = intervals[i]
        splitted_complex_filepath = splitted_complex_filepaths[i]
        splitted_pred_filepath = splitted_pred_filepaths[i]
        splitted_ref_filepath = splitted_ref_filepaths[i]
        scores = get_simplification_scores(splitted_complex_filepath, splitted_pred_filepath, [splitted_ref_filepath])
        row_name = f'{simplifier.__name__}_{interval[0]}_{interval[1]}'
        df_bins = df_append_row(df_bins, scores, row_name)
    return df_bins


def get_markdown_scores(simplifier):
    '''Return a markdown formatted string of turkcorpus and wikilarge scores'''
    df_scores = pd.DataFrame()
    for phase in ['valid', 'test']:
        turkcorpus_scores = evaluate_simplifier_on_turkcorpus(simplifier, phase=phase)
        df_scores = df_append_row(df_scores, turkcorpus_scores, f'Turkcorpus ({phase})')
        # wikilarge_scores = evaluate_simplifier_on_wikilarge(simplifier, phase=phase)
        # df_scores = df_append_row(df_scores, wikilarge_scores, f'Wikilarge ({phase})')
    scores_table = tabulate(df_scores, headers='keys', tablefmt='pipe')
    return f'## Scores and metrics  \n{scores_table}  \n\n'


def get_markdown_scores_by_sentence_length(simplifier):
    '''Return a markdown formatted table string of scores broken down by sentence length'''
    df_bins = evaluate_simplifier_by_sentence_length(simplifier)
    scores_table = tabulate(df_bins, headers='keys', tablefmt='pipe')
    return f'## Wikilarge scores broken down by sentence length:  \n{scores_table}  \n\n'


def get_markdown_report(simplifier):
    text = f'# {simplifier.__name__}  \n'
    text += get_markdown_scores(simplifier)
    text += get_markdown_scores_by_sentence_length(simplifier)
    text += evaluate_simplifier_qualitatively(simplifier)
    text += get_sanity_check_text(simplifier)
    return text
