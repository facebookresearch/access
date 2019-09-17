from functools import lru_cache
import re
from string import punctuation

import nltk
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


def to_sentences(text, language='english'):
    tokenizer = nltk.data.load(f'tokenizers/punkt/{language}.pickle')
    return tokenizer.tokenize(text)


def count_sentences(text, language='english'):
    return len(to_sentences(text, language))


# Adapted from the following scripts:
# https://github.com/XingxingZhang/dress/blob/master/dress/scripts/readability/syllables_en.py
# https://github.com/nltk/nltk_contrib/blob/master/nltk_contrib/readability/syllables_en.py
"""
Fallback syllable counter
This is based on the algorithm in Greg Fast's perl module
Lingua::EN::Syllable.
"""

specialSyllables_en = """tottered 2
chummed 1
peeped 1
moustaches 2
shamefully 3
messieurs 2
satiated 4
sailmaker 4
sheered 1
disinterred 3
propitiatory 6
bepatched 2
particularized 5
caressed 2
trespassed 2
sepulchre 3
flapped 1
hemispheres 3
pencilled 2
motioned 2
poleman 2
slandered 2
sombre 2
etc 4
sidespring 2
mimes 1
effaces 2
mr 2
mrs 2
ms 1
dr 2
st 1
sr 2
jr 2
truckle 2
foamed 1
fringed 2
clattered 2
capered 2
mangroves 2
suavely 2
reclined 2
brutes 1
effaced 2
quivered 2
h'm 1
veriest 3
sententiously 4
deafened 2
manoeuvred 3
unstained 2
gaped 1
stammered 2
shivered 2
discoloured 3
gravesend 2
60 2
lb 1
unexpressed 3
greyish 2
unostentatious 5
"""

fallback_cache = {}

fallback_subsyl = ["cial", "tia", "cius", "cious", "gui", "ion", "iou", "sia$", ".ely$"]

fallback_addsyl = [
    "ia", "riet", "dien", "iu", "io", "ii", "[aeiouy]bl$", "mbl$", "[aeiou]{3}", "^mc", "ism$",
    "(.)(?!\\1)([aeiouy])\\2l$", "[^l]llien", "^coad.", "^coag.", "^coal.", "^coax.",
    "(.)(?!\\1)[gq]ua(.)(?!\\2)[aeiou]", "dnt$"
]

# Compile our regular expressions
for i in range(len(fallback_subsyl)):
    fallback_subsyl[i] = re.compile(fallback_subsyl[i])
for i in range(len(fallback_addsyl)):
    fallback_addsyl[i] = re.compile(fallback_addsyl[i])


def _normalize_word(word):
    return word.strip().lower()


# Read our syllable override file and stash that info in the cache
for line in specialSyllables_en.splitlines():
    line = line.strip()
    if line:
        toks = line.split()
        assert len(toks) == 2
        fallback_cache[_normalize_word(toks[0])] = int(toks[1])


@lru_cache(maxsize=1)
def get_spacy_model():
    return spacy.load('en_core_web_sm')  # python -m spacy download en_core_web_sm


@lru_cache(maxsize=10**6)
def spacy_process(text):
    return get_spacy_model()(str(text))
