# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from abc import ABC
from functools import wraps, lru_cache
import hashlib
from pathlib import Path
import dill as pickle
import re
import shutil

from nevergrad.instrumentation import var
import numpy as np
import sentencepiece as spm

from access.feature_extraction import (get_lexical_complexity_score, get_levenshtein_similarity,
                                       get_dependency_tree_depth)
from access.resources.paths import VARIOUS_DIR, get_data_filepath
from access.utils.helpers import (write_lines_in_parallel, yield_lines_in_parallel, add_dicts, get_default_args,
                                  get_temp_filepath, safe_division, count_lines)

SPECIAL_TOKEN_REGEX = r'<[a-zA-Z\-_\d\.]+>'
PREPROCESSORS_REGISTRY = {}


def get_preprocessor_by_name(preprocessor_name):
    return PREPROCESSORS_REGISTRY[preprocessor_name]


def get_preprocessors(preprocessor_kwargs):
    preprocessors = []
    for preprocessor_name, kwargs in preprocessor_kwargs.items():
        preprocessors.append(get_preprocessor_by_name(preprocessor_name)(**kwargs))
    return preprocessors


def extract_special_tokens(sentence):
    '''Remove any number of token at the beginning of the sentence'''
    match = re.match(fr'(^(?:{SPECIAL_TOKEN_REGEX} *)+) *(.*)$', sentence)
    if match is None:
        return '', sentence
    special_tokens, sentence = match.groups()
    return special_tokens.strip(), sentence


def remove_special_tokens(sentence):
    return extract_special_tokens(sentence)[1]


def store_args(constructor):
    @wraps(constructor)
    def wrapped(self, *args, **kwargs):
        if not hasattr(self, 'args') or not hasattr(self, 'kwargs'):
            # TODO: Default args are not overwritten if provided as args
            self.args = args
            self.kwargs = add_dicts(get_default_args(constructor), kwargs)
        return constructor(self, *args, **kwargs)

    return wrapped


def dump_preprocessors(preprocessors, dir_path):
    with open(Path(dir_path) / 'preprocessors.pickle', 'wb') as f:
        pickle.dump(preprocessors, f)


def load_preprocessors(dir_path):
    path = Path(dir_path) / 'preprocessors.pickle'
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


class AbstractPreprocessor(ABC):
    def __init_subclass__(cls, **kwargs):
        '''Register all children in registry'''
        super().__init_subclass__(**kwargs)
        PREPROCESSORS_REGISTRY[cls.__name__] = cls

    def __repr__(self):
        args = getattr(self, 'args', ())
        kwargs = getattr(self, 'kwargs', {})
        args_repr = [repr(arg) for arg in args]
        kwargs_repr = [f'{k}={repr(v)}' for k, v in sorted(kwargs.items(), key=lambda kv: kv[0])]
        args_kwargs_str = ', '.join(args_repr + kwargs_repr)
        return f'{self.__class__.__name__}({args_kwargs_str})'

    def get_hash_string(self):
        return self.__class__.__name__

    def get_hash(self):
        return hashlib.md5(self.get_hash_string().encode()).hexdigest()

    def get_nevergrad_variables(self):
        return {}

    @property
    def prefix(self):
        return self.__class__.__name__.replace('Preprocessor', '')

    def fit(self, complex_filepath, simple_filepath):
        pass

    def encode_sentence(self, sentence, encoder_sentence=None):
        raise NotImplementedError

    def decode_sentence(self, sentence, encoder_sentence=None):
        raise NotImplementedError

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        if complex_sentence is not None:
            complex_sentence = self.encode_sentence(complex_sentence)
        if simple_sentence is not None:
            simple_sentence = self.encode_sentence(simple_sentence)
        return complex_sentence, simple_sentence

    def encode_file(self, input_filepath, output_filepath, encoder_filepath=None):
        if encoder_filepath is None:
            # We will use an empty temporary file which will yield None for each line
            encoder_filepath = get_temp_filepath(create=True)
        with open(output_filepath, 'w') as f:
            for input_line, encoder_line in yield_lines_in_parallel([input_filepath, encoder_filepath], strict=False):
                f.write(self.encode_sentence(input_line, encoder_line) + '\n')

    def decode_file(self, input_filepath, output_filepath, encoder_filepath=None):
        if encoder_filepath is None:
            # We will use an empty temporary file which will yield None for each line
            encoder_filepath = get_temp_filepath(create=True)
        with open(output_filepath, 'w') as f:
            for encoder_sentence, input_sentence in yield_lines_in_parallel([encoder_filepath, input_filepath],
                                                                            strict=False):
                decoded_sentence = self.decode_sentence(input_sentence, encoder_sentence=encoder_sentence)
                f.write(decoded_sentence + '\n')

    def encode_file_pair(self, complex_filepath, simple_filepath, output_complex_filepath, output_simple_filepath):
        '''Jointly encode a complex file and a simple file (can be aligned or not)'''
        with write_lines_in_parallel([output_complex_filepath, output_simple_filepath], strict=False) as output_files:
            for complex_line, simple_line in yield_lines_in_parallel([complex_filepath, simple_filepath], strict=False):
                output_files.write(self.encode_sentence_pair(complex_line, simple_line))


class ComposedPreprocessor(AbstractPreprocessor):
    @store_args
    def __init__(self, preprocessors, sort=False):
        if preprocessors is None:
            preprocessors = []
        if sort:
            # Make sure preprocessors are always in the same order
            preprocessors = sorted(preprocessors, key=lambda preprocessor: preprocessor.__class__.__name__)
        self.preprocessors = preprocessors

    def get_hash_string(self):
        preprocessors_hash_strings = [preprocessor.get_hash_string() for preprocessor in self.preprocessors]
        return f'ComposedPreprocessor(preprocessors={preprocessors_hash_strings})'

    def get_suffix(self):
        return '_'.join([p.prefix.lower() for p in self.preprocessors])

    def fit(self, complex_filepath, simple_filepath):
        for preprocessor in self.preprocessors:
            pass

    def encode_sentence(self, sentence, encoder_sentence=None):
        for preprocessor in self.preprocessors:
            sentence = preprocessor.encode_sentence(sentence, encoder_sentence)
        return sentence

    def decode_sentence(self, sentence, encoder_sentence=None):
        for preprocessor in self.preprocessors:
            sentence = preprocessor.decode_sentence(sentence, encoder_sentence)
        return sentence

    def encode_file(self, input_filepath, output_filepath, encoder_filepath=None):
        for preprocessor in self.preprocessors:
            intermediary_output_filepath = get_temp_filepath()
            preprocessor.encode_file(input_filepath, intermediary_output_filepath, encoder_filepath)
            input_filepath = intermediary_output_filepath
        shutil.copyfile(input_filepath, output_filepath)

    def decode_file(self, input_filepath, output_filepath, encoder_filepath=None):
        for preprocessor in self.preprocessors:
            intermediary_output_filepath = get_temp_filepath()
            preprocessor.decode_file(input_filepath, intermediary_output_filepath, encoder_filepath)
            input_filepath = intermediary_output_filepath
        shutil.copyfile(input_filepath, output_filepath)

    def encode_file_pair(self, complex_filepath, simple_filepath, output_complex_filepath, output_simple_filepath):
        for preprocessor in self.preprocessors:
            intermediary_output_complex_filepath = get_temp_filepath()
            intermediary_output_simple_filepath = get_temp_filepath()
            preprocessor.encode_file_pair(complex_filepath, simple_filepath, intermediary_output_complex_filepath,
                                          intermediary_output_simple_filepath)
            complex_filepath = intermediary_output_complex_filepath
            simple_filepath = intermediary_output_simple_filepath
        shutil.copyfile(complex_filepath, output_complex_filepath)
        shutil.copyfile(simple_filepath, output_simple_filepath)

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        for preprocessor in self.preprocessors:
            complex_sentence, simple_sentence = preprocessor.encode_sentence_pair(complex_sentence, simple_sentence)
        return complex_sentence, simple_sentence


class FeaturePreprocessor(AbstractPreprocessor):
    '''Prepend a computed feature at the beginning of the sentence'''
    @store_args
    def __init__(self, feature_name, get_feature_value, get_target_feature_value, bucket_size=0.05, noise_std=0):
        self.get_feature_value = get_feature_value
        self.get_target_feature_value = get_target_feature_value
        self.bucket_size = bucket_size
        self.noise_std = noise_std
        self.feature_name = feature_name.upper()

    def get_hash_string(self):
        return (f'{self.__class__.__name__}(feature_name={repr(self.feature_name)}, bucket_size={self.bucket_size},'
                f'noise_std={self.noise_std})')

    def bucketize(self, value):
        '''Round value to bucket_size to reduce the number of different values'''
        return round(round(value / self.bucket_size) * self.bucket_size, 10)

    def add_noise(self, value):
        return value + np.random.normal(0, self.noise_std)

    def get_feature_token(self, feature_value):
        return f'<{self.feature_name}_{feature_value}>'

    def encode_sentence(self, sentence, encoder_sentence=None):
        desired_feature = self.bucketize(self.get_target_feature_value(remove_special_tokens(sentence)))
        return f'{self.get_feature_token(desired_feature)} {sentence}'

    def decode_sentence(self, sentence, encoder_sentence=None):
        return sentence

    def encode_sentence_pair(self, complex_sentence, simple_sentence):
        feature = self.bucketize(
            self.add_noise(
                self.get_feature_value(remove_special_tokens(complex_sentence),
                                       remove_special_tokens(simple_sentence))))
        return f'{self.get_feature_token(feature)} {complex_sentence}', simple_sentence


class LevenshteinPreprocessor(FeaturePreprocessor):
    @store_args
    def __init__(self, target_ratio=0.8, bucket_size=0.05, noise_std=0):
        self.target_ratio = target_ratio
        super().__init__(self.prefix.upper(), self.get_feature_value, self.get_target_feature_value, bucket_size,
                         noise_std)

    def get_nevergrad_variables(self):
        return {'target_ratio': var.OrderedDiscrete(np.arange(0.4, 1 + 1e-6, self.bucket_size))}

    def get_feature_value(self, complex_sentence, simple_sentence):
        return get_levenshtein_similarity(complex_sentence, simple_sentence)

    def get_target_feature_value(self, complex_sentence):
        return self.target_ratio


class RatioPreprocessor(FeaturePreprocessor):
    @store_args
    def __init__(self, feature_extractor, target_ratio=0.8, bucket_size=0.05, noise_std=0):
        self.feature_extractor = feature_extractor
        self.target_ratio = target_ratio
        super().__init__(self.prefix.upper(), self.get_feature_value, self.get_target_feature_value, bucket_size,
                         noise_std)

    def get_nevergrad_variables(self):
        return {'target_ratio': var.OrderedDiscrete(np.arange(0.4, 1.4 + 1e-6, self.bucket_size))}

    def get_feature_value(self, complex_sentence, simple_sentence):
        return min(safe_division(self.feature_extractor(simple_sentence), self.feature_extractor(complex_sentence)), 2)

    def get_target_feature_value(self, complex_sentence):
        return self.target_ratio


class LengthRatioPreprocessor(RatioPreprocessor):
    @store_args
    def __init__(self, *args, **kwargs):
        super().__init__(len, *args, **kwargs)


class WordRankRatioPreprocessor(RatioPreprocessor):
    @store_args
    def __init__(self, *args, **kwargs):
        super().__init__(get_lexical_complexity_score, *args, **kwargs)


class DependencyTreeDepthRatioPreprocessor(RatioPreprocessor):
    @store_args
    def __init__(self, *args, **kwargs):
        super().__init__(get_dependency_tree_depth, *args, **kwargs)


class SentencePiecePreprocessor(AbstractPreprocessor):
    @store_args
    def __init__(self, vocab_size=10000, input_filepaths=None):
        self.vocab_size = vocab_size
        self.sentencepiece_model_path = VARIOUS_DIR / f'sentencepiece_model/sentencepiece_model_{self.vocab_size}.model'
        self.input_filepaths = input_filepaths
        if self.input_filepaths is None:
            self.input_filepaths = [
                get_data_filepath('wikilarge', 'train', 'complex'),
                get_data_filepath('wikilarge', 'train', 'simple')
            ]
        self.learn_sentencepiece()

    @property
    @lru_cache(maxsize=1)
    def sp(self):
        '''
        We need to use a property because SentencenPieceProcessor is cannot pickled
        > pickle.dumps(spm.SentencePieceProcessor())
        ----> TypeError: can't pickle SwigPyObject objects
        '''
        sp = spm.SentencePieceProcessor()
        sp.Load(str(self.sentencepiece_model_path))
        return sp

    def get_hash_string(self):
        return f'{self.__class__.__name__}(vocab_size={self.vocab_size})'

    def learn_sentencepiece(self):
        if self.sentencepiece_model_path.exists():
            return
        self.sentencepiece_model_path.parent.mkdir(parents=True, exist_ok=True)
        sentencepiece_model_prefix = self.sentencepiece_model_path.parent / self.sentencepiece_model_path.stem
        args_str = ' '.join([
            f'--input={",".join([str(path) for path in self.input_filepaths])}',
            f'--model_prefix={sentencepiece_model_prefix}',
            f'--vocab_size={self.vocab_size}',
        ])
        max_lines = 10**6
        if sum([count_lines(filepath) for filepath in self.input_filepaths]) > max_lines:
            args_str += f' --input_sentence_size={max_lines} --shuffle_input_sentence=true'
        spm.SentencePieceTrainer.Train(args_str)

    def fit(self, complex_filepath, simple_filepath):
        # Args are not used
        self.learn_sentencepiece()

    def encode_sentence(self, sentence, encoder_sentence=None):
        # TODO: Do we really need to extract the tokens
        special_tokens, sentence = extract_special_tokens(sentence)
        encoded_sentence = ' '.join(self.sp.EncodeAsPieces(sentence))
        if special_tokens != '':
            encoded_sentence = f'{special_tokens} {encoded_sentence}'
        return encoded_sentence

    def decode_sentence(self, sentence, encoder_sentence=None):
        return self.sp.DecodePieces(sentence.split(' '))
