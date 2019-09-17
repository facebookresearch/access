# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from functools import lru_cache
import re
from string import punctuation

from nltk.tokenize.nist import NISTTokenizer
from nltk.corpus import stopwords as nltk_stopwords
import spacy

# TODO: #language_specific
stopwords = set(nltk_stopwords.words('english'))


@lru_cache(maxsize=100)  # To speed up subsequent calls
def word_tokenize(sentence):
    tokenizer = NISTTokenizer()
    sentence = ' '.join(tokenizer.tokenize(sentence))
    # Rejoin special tokens that where tokenized by error: e.g. "<PERSON_1>" -> "< PERSON _ 1 >"
    for match in re.finditer(r'< (?:[A-Z]+ _ )+\d+ >', sentence):
        sentence = sentence.replace(match.group(), ''.join(match.group().split()))
    return sentence


def to_words(sentence):
    return sentence.split()


def remove_punctuation_characters(text):
    return ''.join([char for char in text if char not in punctuation])


@lru_cache(maxsize=1000)
def is_punctuation(word):
    return remove_punctuation_characters(word) == ''


@lru_cache(maxsize=100)
def remove_punctuation_tokens(text):
    return ' '.join([w for w in to_words(text) if not is_punctuation(w)])


def remove_stopwords(text):
    return ' '.join([w for w in to_words(text) if w.lower() not in stopwords])


@lru_cache(maxsize=1)
def get_spacy_model():
    model = 'en_core_web_sm'
    if not spacy.util.is_package(model):
        spacy.cli.download(model)
        spacy.cli.link(model, model, force=True, model_path=spacy.util.get_package_path(model))
    return spacy.load(model)  # python -m spacy download en_core_web_sm`


@lru_cache(maxsize=10**6)
def spacy_process(text):
    return get_spacy_model()(str(text))
