from functools import lru_cache

import Levenshtein
import numpy as np

from ts.resources.paths import FASTTEXT_EMBEDDINGS_PATH
from ts.text import count_sentences, to_words, remove_punctuation_tokens, remove_stopwords, spacy_process
from ts.utils.helpers import safe_division, yield_lines


@lru_cache(maxsize=1)
def get_word2rank(vocab_size=np.inf):
    # TODO: Decrease vocab size or load from smaller file
    word2rank = {}
    line_generator = yield_lines(FASTTEXT_EMBEDDINGS_PATH)
    next(line_generator)  # Skip the first line (header)
    for i, line in enumerate(line_generator):
        if (i + 1) > vocab_size:
            break
        word = line.split(' ')[0]
        word2rank[word] = i
    return word2rank


def get_rank(word):
    return get_word2rank().get(word, len(get_word2rank()))


def get_log_rank(word):
    return np.log(1 + get_rank(word))


def get_lexical_complexity_score(sentence):
    words = to_words(remove_stopwords(remove_punctuation_tokens(sentence)))
    words = [word for word in words if word in get_word2rank()]
    if len(words) == 0:
        return np.log(1 + len(get_word2rank()))  # TODO: This is completely arbitrary
    return np.quantile([get_log_rank(word) for word in words], 0.75)


def count_characters(sentence):
    return len(sentence)


def compression_ratio(complex_sentence, simple_sentence):
    return safe_division(count_characters(simple_sentence), count_characters(complex_sentence))


def get_levenshtein_similarity(complex_sentence, simple_sentence):
    return Levenshtein.ratio(complex_sentence, simple_sentence)


def get_levenshtein_distance(complex_sentence, simple_sentence):
    return 1 - get_levenshtein_similarity(complex_sentence, simple_sentence)


def count_sentence_splits(complex_sentence, simple_sentence):
    return safe_division(count_sentences(simple_sentence), count_sentences(complex_sentence))


def get_dependency_tree_depth(sentence):
    def get_subtree_depth(node):
        if len(list(node.children)) == 0:
            return 0
        return 1 + max([get_subtree_depth(child) for child in node.children])

    tree_depths = [get_subtree_depth(spacy_sentence.root) for spacy_sentence in spacy_process(sentence).sents]
    if len(tree_depths) == 0:
        return 0
    return max(tree_depths)
