# TODO: Move to utils/training.py
from functools import wraps
import time

from ts.resources.paths import REPO_DIR
from ts.utils.helpers import run_command


def print_method_name(func):
    '''Decorator to print method name for logging purposes'''
    @wraps(func)  # To preserve the name and path for pickling purposes
    def wrapped_func(*args, **kwargs):
        print(f"method_name='{func.__name__}'")
        return func(*args, **kwargs)

    return wrapped_func


def print_args(func):
    '''Decorator to print arguments of method for logging purposes'''
    @wraps(func)  # To preserve the name and path for pickling purposes
    def wrapped_func(*args, **kwargs):
        print(f'args={args}')
        print(f'kwargs={kwargs}')
        return func(*args, **kwargs)

    return wrapped_func


def print_result(func):
    '''Decorator to print result of method for logging purposes'''
    @wraps(func)  # To preserve the name and path for pickling purposes
    def wrapped_func(*args, **kwargs):
        result = func(*args, **kwargs)
        print(f'result={result}')
        return result

    return wrapped_func


def print_running_time(func):
    '''Decorator to print running time of method for logging purposes'''
    @wraps(func)  # To preserve the name and path for pickling purposes
    def wrapped_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        print(f'running_time={time.time() - start_time}')
        return result

    return wrapped_func


def get_current_commit_hash():
    return run_command(f'git --git-dir="{REPO_DIR}/.git" describe --always', mute=True)


def clone_repo(destination_dir):
    # TODO: Commit unstaged changes
    run_command(f'git clone {REPO_DIR}/.git {destination_dir}')
