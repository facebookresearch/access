from itertools import product
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_DIR / 'experiments'
RESOURCES_DIR = REPO_DIR / 'resources'
CACHE_DIR = EXP_DIR / 'cache'
DATASETS_DIR = RESOURCES_DIR / 'datasets'
VARIOUS_DIR = RESOURCES_DIR / 'various'
MODELS_DIR = RESOURCES_DIR / 'models'
TOOLS_DIR = RESOURCES_DIR / 'tools'
FAIRSEQ_CACHED_CHECKPOINTS_DIR = EXP_DIR / 'fairseq_cached_checkpoints'
# TODO: Move this to setup or add the folders to the git repo
for dir_path in [DATASETS_DIR, VARIOUS_DIR, MODELS_DIR, TOOLS_DIR, CACHE_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)
FASTTEXT_EMBEDDINGS_PATH = Path(VARIOUS_DIR) / 'fasttext-vectors/wiki.en.vec'

LANGUAGES = ['complex', 'simple']
PHASES = ['train', 'valid', 'test']


def get_dataset_dir(dataset):
    return DATASETS_DIR / dataset


def get_data_filepath(dataset, phase, language, i=None):
    suffix = ''  # Create suffix e.g. for multiple references
    if i is not None:
        suffix = f'.{i}'
    filename = f'{dataset}.{phase}.{language}{suffix}'
    return get_dataset_dir(dataset) / filename


def get_filepaths_dict(dataset):
    return {(phase, language): get_data_filepath(dataset, phase, language)
            for phase, language in product(PHASES, LANGUAGES)}
