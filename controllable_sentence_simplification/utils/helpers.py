import ast
from contextlib import contextmanager, AbstractContextManager
from fcntl import flock, LOCK_EX, LOCK_UN
import inspect
import io
from itertools import zip_longest
from pathlib import Path
import select
import shutil
from subprocess import Popen, PIPE, CalledProcessError
import sys
import tempfile
import time

import numpy as np


def run_command(cmd, mute=False):
    def readline_with_timeout(input_stream, timeout=0.1):
        '''Avoids handing indefinitely when calling readline()
        https://stackoverflow.com/questions/10756383/timeout-on-subprocess-readline-in-python'''
        poll_obj = select.poll()
        poll_obj.register(input_stream, select.POLLIN)
        start = time.time()
        while (time.time() - start) < timeout:
            poll_result = poll_obj.poll(0)
            if poll_result:
                return input_stream.readline()
        return ''

    def get_available_output(input_stream):
        output = ''
        while True:
            line = readline_with_timeout(input_stream, timeout=0.1)
            if line == '':
                break
            output += line
        return output

    def read_and_print(input_stream, output_stream):
        output = get_available_output(input_stream)
        if not mute:
            print(output, file=output_stream, end='', flush=True)
        return output

    # Inspired from subprocess.run() source
    # HACK: shell=True is not secure
    with Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True, encoding='utf-8') as process:
        try:
            stdout = ''
            stderr = ''
            while True:
                stdout += read_and_print(process.stdout, sys.stdout)
                stderr += read_and_print(process.stderr, sys.stderr)
                if process.poll() is not None:
                    break
            # Read remaining output in case there is some
            stdout += read_and_print(process.stdout, sys.stdout)
            stderr += read_and_print(process.stderr, sys.stderr)
        except Exception:
            if mute:  # Print errors that could have been muted
                print(stderr, file=sys.stderr)
            print(get_available_output(process.stderr), file=sys.stderr)
            process.kill()
            process.wait()
            raise
        retcode = process.poll()
        if retcode:
            print(stderr, file=sys.stderr)
            raise CalledProcessError(retcode, process.args, output=stdout, stderr=stderr)
    return stdout.strip()


@contextmanager
def open_files(filepaths, mode='r'):
    files = []
    try:
        files = [Path(filepath).open(mode) for filepath in filepaths]
        yield files
    finally:
        [f.close() for f in files]


def yield_lines_in_parallel(filepaths, strip=True, strict=True, n_lines=float('inf')):
    assert type(filepaths) == list
    with open_files(filepaths) as files:
        for i, parallel_lines in enumerate(zip_longest(*files)):
            if i >= n_lines:
                break
            if None in parallel_lines:
                assert not strict, f'Files don\'t have the same number of lines: {filepaths}, use strict=False'
            if strip:
                parallel_lines = [l.rstrip('\n') if l is not None else None for l in parallel_lines]
            yield parallel_lines


class FilesWrapper:
    '''Write to multiple open files at the same time'''
    def __init__(self, files, strict=True):
        self.files = files
        self.strict = strict  # Whether to raise an exception when a line is None

    def write(self, lines):
        assert len(lines) == len(self.files)
        for line, f in zip(lines, self.files):
            if line is None:
                assert not self.strict
                continue
            f.write(line.rstrip('\n') + '\n')


@contextmanager
def write_lines_in_parallel(filepaths, strict=True):
    with open_files(filepaths, 'w') as files:
        yield FilesWrapper(files, strict=strict)


def write_lines(lines, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open('w') as f:
        for line in lines:
            f.write(line + '\n')


def yield_lines(filepath, n_lines=float('inf'), prop=1):
    if prop < 1:
        assert n_lines == float('inf')
        n_lines = int(prop * count_lines(filepath))
    with open(filepath, 'r') as f:
        for i, l in enumerate(f):
            if i >= n_lines:
                break
            yield l.rstrip('\n')


def read_lines(filepath, n_lines=float('inf'), prop=1):
    return list(yield_lines(filepath, n_lines, prop))


def read_file(filepath):
    with Path(filepath).open('r') as f:
        return f.read()


def count_lines(filepath):
    n_lines = 0
    with Path(filepath).open() as f:
        for l in f:
            n_lines += 1
    return n_lines


@contextmanager
def open_with_lock(filepath, mode):
    with open(filepath, mode) as f:
        flock(f, LOCK_EX)
        yield f
        flock(f, LOCK_UN)


def get_lockfile_path(path):
    path = Path(path)
    if path.is_dir():
        return path / '.lockfile'
    if path.is_file():
        return path.parent / f'.{path.name}.lockfile'


@contextmanager
def lock_file(filepath):
    '''Lock file foo.txt by creating a lock on .foo.txt.lock'''
    # TODO: do we really need to create an additional file for locking ?
    filepath = Path(filepath)
    assert filepath.exists(), f'File does not exists: {filepath}'
    lockfile_path = get_lockfile_path(filepath)
    with open_with_lock(lockfile_path, 'w'):
        yield


@contextmanager
def lock_directory(dir_path):
    # TODO: Locking a directory should lock all files in that directory
    # Right now if we lock foo/, someone else can lock foo/bar.txt
    # TODO: Nested with lock_directory() should not be blocking
    assert Path(dir_path).exists(), f'Directory does not exists: {dir_path}'
    lockfile_path = get_lockfile_path(dir_path)
    with open_with_lock(lockfile_path, 'w'):
        yield


def safe_division(a, b):
    if b == 0:
        return 0
    return a / b


def harmonic_mean(values, coefs=None):
    if 0 in values:
        return 0
    values = np.array(values)
    if coefs is None:
        coefs = np.ones(values.shape)
    values = np.array(values)
    coefs = np.array(coefs)
    return np.sum(coefs) / np.dot(coefs, 1 / values)


def safe_ast_literal_eval(expression):
    try:
        return ast.literal_eval(expression)
    except (SyntaxError, ValueError):
        return expression


@contextmanager
def mute(mute_stdout=True, mute_stderr=True):
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    if mute_stdout:
        sys.stdout = io.StringIO()
    if mute_stderr:
        sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr


@contextmanager
def log_stdout(filepath, mute_stdout=False):
    '''Context manager to write both to stdout and to a file'''
    class MultipleStreamsWriter:
        def __init__(self, streams):
            self.streams = streams

        def write(self, message):
            for stream in self.streams:
                stream.write(message)

        def flush(self):
            for stream in self.streams:
                stream.flush()

    save_stdout = sys.stdout
    log_file = open(filepath, 'w')
    if mute_stdout:
        sys.stdout = MultipleStreamsWriter([log_file])  # Write to file only
    else:
        sys.stdout = MultipleStreamsWriter([save_stdout, log_file])  # Write to both stdout and file
    try:
        yield
    finally:
        sys.stdout = save_stdout
        log_file.close()


def add_dicts(*dicts):
    return {k: v for dic in dicts for k, v in dic.items()}


def get_default_args(func):
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_allowed_kwargs(func, *args, **kwargs):
    expected_args = inspect.getargspec(func).args
    allowed_kwargs = expected_args[len(args):]
    return {k: v for k, v in kwargs.items() if k in allowed_kwargs}


def get_line_lengths(filepath):
    line_lengths = []
    with open(filepath, 'r') as f:
        for line in f:
            line_lengths.append(len(line.rstrip('\n')))
    return line_lengths


class SkipWithBlock(Exception):
    pass


class create_directory_or_skip(AbstractContextManager):
    '''Context manager for creating a new directory (with rollback and skipping with block if exists)

    In order to skip the execution of the with block if the dataset already exists, this context manager uses deep
    magic from https://stackoverflow.com/questions/12594148/skipping-execution-of-with-block
    '''
    def __init__(self, dir_path, overwrite=False):
        self.dir_path = Path(dir_path)
        self.overwrite = overwrite

    def __enter__(self):
        if self.dir_path.exists():
            self.directory_lock = lock_directory(self.dir_path)
            self.directory_lock.__enter__()
            files_in_directory = list(self.dir_path.iterdir())
            if set(files_in_directory) in [set([]), set([self.dir_path / '.lockfile'])]:
                # TODO: Quick hack to remove empty directories
                self.directory_lock.__exit__(None, None, None)
                print(f'Removing empty directory {self.dir_path}')
                shutil.rmtree(self.dir_path)
            else:
                # Deep magic hack to skip the execution of the code inside the with block
                # We set the trace to a dummy function
                sys.settrace(lambda *args, **keys: None)
                # Get the calling frame (sys._getframe(0) is the current frame)
                frame = sys._getframe(1)
                # Set the calling frame's trace to the one that raises the special exception
                frame.f_trace = self.trace
                return
        print(f'Creating {self.dir_path}...')
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.directory_lock = lock_directory(self.dir_path)
        self.directory_lock.__enter__()

    def trace(self, frame, event, arg):
        # This method is called when a new local scope is entered, i.e. right when the code in the with block begins
        # The exception will therefore be caught by the __exit__()
        raise SkipWithBlock()

    def __exit__(self, type, value, traceback):
        self.directory_lock.__exit__(type, value, traceback)
        if type is not None:
            if issubclass(type, SkipWithBlock):
                return True  # Suppress special SkipWithBlock exception
            if issubclass(type, BaseException):
                # Rollback
                print(f'Error: Rolling back creation of directory {self.dir_path}')
                shutil.rmtree(self.dir_path)
                return False  # Reraise the exception


def get_temp_filepath(create=False):
    temp_filepath = Path(tempfile.mkstemp()[1])
    if not create:
        temp_filepath.unlink()
    return temp_filepath


def get_temp_filepaths(n_filepaths, create=False):
    return [get_temp_filepath(create=create) for _ in range(n_filepaths)]


def delete_files(filepaths):
    for filepath in filepaths:
        filepath = Path(filepath)
        assert filepath.is_file()
        filepath.unlink()
