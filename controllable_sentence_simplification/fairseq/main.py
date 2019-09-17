from collections import defaultdict
from functools import lru_cache
import shutil

from nevergrad.instrumentation import Instrumentation
from nevergrad.optimization import optimizerlib
import re

from ts.evaluation.general import evaluate_simplifier_on_turkcorpus, get_markdown_report
from ts.evaluation.utils import combine_metrics
from ts.fairseq.base import (fairseq_preprocess, fairseq_train, fairseq_generate, get_fairseq_exp_dir,
                             cache_fairseq_train)
from ts.resources.datasets import has_lines_in_common
from ts.preprocessors import get_preprocessors, get_preprocessor_by_name
from ts.resources.datasets import create_preprocessed_dataset
from ts.resources.paths import get_data_filepath, get_dataset_dir
from ts.simplifiers import get_fairseq_simplifier, get_preprocessed_simplifier
from ts.utils.training import (print_method_name, print_args, print_result, print_running_time, clone_repo,
                               get_current_commit_hash)
from ts.utils.helpers import get_allowed_kwargs, mute


def write_fairseq_markdown_report(simplifier, exp_dir):
    markdown_report_path = exp_dir / 'markdown_report.md'
    with open(markdown_report_path, 'w') as f:
        with mute():
            f.write(get_markdown_report(simplifier))
    print(f'markdown_report_path="{markdown_report_path}"')


def check_dataset(dataset):
    # Sanity check with evaluation dataset
    assert not has_lines_in_common(get_data_filepath(dataset, 'train', 'complex'),
                                   get_data_filepath('turkcorpus', 'valid', 'complex'))
    assert not has_lines_in_common(get_data_filepath(dataset, 'train', 'complex'),
                                   get_data_filepath('turkcorpus', 'test', 'complex'))


def prepare_exp_dir():
    exp_dir = get_fairseq_exp_dir()
    if exp_dir.exists():
        # Remove exp dir to prevent conflicts with requeue and non deterministic args
        # https://github.com/fairinternal/dfoptim/issues/126 #private
        shutil.rmtree(exp_dir)
    exp_dir.mkdir()
    # Save source code
    clone_repo(exp_dir / f'text_simplification-{get_current_commit_hash()}')
    return exp_dir


def get_simplifier(exp_dir, preprocessors_kwargs, generate_kwargs):
    # TODO: Take kwargs as input and separate between get_preprocessors kwargs and generate_kwargs
    preprocessors = get_preprocessors(preprocessors_kwargs)
    simplifier = get_fairseq_simplifier(exp_dir, **generate_kwargs)
    return get_preprocessed_simplifier(simplifier, preprocessors=preprocessors)


def find_best_parametrization(exp_dir, metrics_coefs, preprocessors_kwargs, parametrization_budget=64):
    @lru_cache()
    def evaluate_parametrization(**instru_kwargs):
        # Note that we use default generate kwargs instead of provided one because they are faster
        preprocessors_kwargs = instru_kwargs_to_preprocessors_kwargs(instru_kwargs)
        simplifier = get_simplifier(exp_dir, preprocessors_kwargs=preprocessors_kwargs, generate_kwargs={})
        scores = evaluate_simplifier_on_turkcorpus(simplifier, phase='valid')
        return combine_metrics(scores['BLEU'], scores['SARI'], scores['FKGL'], metrics_coefs)

    def preprocessors_kwargs_to_instru_kwargs(preprocessors_kwargs):
        instru_kwargs = {}
        for preprocessor_name, preprocessor_kwargs in preprocessors_kwargs.items():
            assert '_' not in preprocessor_name
            preprocessor = get_preprocessor_by_name(preprocessor_name)(**preprocessor_kwargs)
            # First we set the values from preprocessors_kwargs which are constant
            for kwarg_name, kwarg_value in preprocessor_kwargs.items():
                instru_kwargs[f'{preprocessor_name}_{kwarg_name}'] = kwarg_value
            # Then we overwrite some of these values with nevergrad variables when necessary
            for kwarg_name, kwarg_value in preprocessor.get_nevergrad_variables().items():
                instru_kwargs[f'{preprocessor_name}_{kwarg_name}'] = kwarg_value
        return instru_kwargs

    def instru_kwargs_to_preprocessors_kwargs(instru_kwargs):
        preprocessors_kwargs = defaultdict(dict)
        for key, value in instru_kwargs.items():
            preprocessor_name, kwarg_name = re.match(r'([a-zA-Z0-9]+)_([a-z0-9_]+)', key).groups()
            preprocessors_kwargs[preprocessor_name][kwarg_name] = value
        return dict(preprocessors_kwargs)

    instru_kwargs = preprocessors_kwargs_to_instru_kwargs(preprocessors_kwargs)
    instru = Instrumentation(**instru_kwargs)
    if instru.dimension == 0:
        return preprocessors_kwargs
    # No need to search a lot when there is only a few parameters
    parametrization_budget = min(32**instru.dimension, parametrization_budget)
    optimizer = optimizerlib.ScrHammersleySearch(instrumentation=instru, budget=parametrization_budget, num_workers=1)
    recommendation = optimizer.optimize(evaluate_parametrization, verbosity=0)
    return instru_kwargs_to_preprocessors_kwargs(recommendation.kwargs)


def check_and_resolve_args(kwargs):
    if kwargs.get('diverse_beam_groups_ratio', None) is not None:
        diverse_beam_groups = max(int(kwargs['beam'] * kwargs['diverse_beam_groups_ratio']), 1)
        print(f'diverse_beam_groups={diverse_beam_groups}')
        assert kwargs['beam'] % diverse_beam_groups == 0
        kwargs['diverse_beam_groups'] = diverse_beam_groups
    else:
        diverse_beam_groups = None
    return kwargs


@print_method_name
@print_args
@print_result
@print_running_time
def fairseq_train_and_evaluate(dataset,
                               metrics_coefs=[1, 1, 1],
                               cache=False,
                               parametrization_budget=64,
                               **kwargs):
    check_dataset(dataset)
    kwargs = check_and_resolve_args(kwargs)
    exp_dir = prepare_exp_dir()
    preprocessors_kwargs = kwargs.get('preprocessors_kwargs', {})
    preprocessors = get_preprocessors(preprocessors_kwargs)
    if len(preprocessors) > 0:
        dataset = create_preprocessed_dataset(dataset, preprocessors, n_jobs=8)
        shutil.copy(get_dataset_dir(dataset) / 'preprocessors.pickle', exp_dir)
    preprocessed_dir = fairseq_preprocess(dataset)
    train_kwargs = get_allowed_kwargs(fairseq_train, preprocessed_dir, exp_dir, **kwargs)
    if cache:
        cache_fairseq_train(fairseq_train(preprocessed_dir, exp_dir=exp_dir, **train_kwargs))
    else:
        fairseq_train(preprocessed_dir, exp_dir=exp_dir, **train_kwargs)
    # Evaluation
    generate_kwargs = get_allowed_kwargs(fairseq_generate, 'complex_filepath', 'pred_filepath', exp_dir, **kwargs)
    recommended_preprocessors_kwargs = find_best_parametrization(exp_dir, metrics_coefs, preprocessors_kwargs,
                                                                 parametrization_budget)
    print(f'recommended_preprocessors_kwargs={recommended_preprocessors_kwargs}')
    simplifier = get_simplifier(exp_dir, recommended_preprocessors_kwargs, generate_kwargs)
    scores = evaluate_simplifier_on_turkcorpus(simplifier, phase='valid')
    print(f'scores={scores}')
    write_fairseq_markdown_report(simplifier, exp_dir)
    score = combine_metrics(scores['BLEU'], scores['SARI'], scores['FKGL'], metrics_coefs)
    return score
