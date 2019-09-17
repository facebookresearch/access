# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import bz2
import gzip
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import time
from urllib.request import urlretrieve
import zipfile

import git
from tqdm import tqdm


def reporthook(count, block_size, total_size):
    # Download progress bar
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size_mb = count * block_size / (1024 * 1024)
    speed = progress_size_mb / duration
    percent = int(count * block_size * 100 / total_size)
    msg = f'\r... {percent}% - {int(progress_size_mb)} MB - {speed:.2f} MB/s - {int(duration)}s'
    sys.stdout.write(msg)


def download(url, destination_path):
    print('Downloading...')
    try:
        urlretrieve(url, destination_path, reporthook)
        sys.stdout.write('\n')
    except (Exception, KeyboardInterrupt, SystemExit):
        print('Rolling back: remove partially downloaded file')
        os.remove(destination_path)
        raise


def download_and_extract(url):
    tmp_dir = Path(tempfile.mkdtemp())
    compressed_filename = url.split('/')[-1]
    compressed_filepath = tmp_dir / compressed_filename
    download(url, compressed_filepath)
    print('Extracting...')
    return extract(compressed_filepath, tmp_dir)


def extract(filepath, output_dir):
    # Infer extract method based on extension
    extensions_to_methods = {
        '.tar.gz': untar,
        '.tar.bz2': untar,
        '.tgz': untar,
        '.zip': unzip,
        '.gz': ungzip,
        '.bz2': unbz2,
    }

    def get_extension(filename, extensions):
        possible_extensions = [ext for ext in extensions if filename.endswith(ext)]
        if len(possible_extensions) == 0:
            raise Exception(f'File {filename} has an unknown extension')
        # Take the longest (.tar.gz should take precedence over .gz)
        return max(possible_extensions, key=lambda ext: len(ext))

    filename = os.path.basename(filepath)
    extension = get_extension(filename, list(extensions_to_methods))
    extract_method = extensions_to_methods[extension]

    # Extract files in a temporary dir then move the extracted item back to
    # the ouput dir in order to get the details of what was extracted
    tmp_extract_dir = tempfile.mkdtemp()
    # Extract
    extract_method(filepath, output_dir=tmp_extract_dir)
    extracted_items = os.listdir(tmp_extract_dir)
    output_paths = []
    for name in extracted_items:
        extracted_path = os.path.join(tmp_extract_dir, name)
        output_path = os.path.join(output_dir, name)
        move_with_overwrite(extracted_path, output_path)
        output_paths.append(output_path)
    return output_paths


def move_with_overwrite(source_path, target_path):
    if os.path.isfile(target_path):
        os.remove(target_path)
    if os.path.isdir(target_path) and os.path.isdir(source_path):
        shutil.rmtree(target_path)
    shutil.move(source_path, target_path)


def untar(compressed_path, output_dir):
    with tarfile.open(compressed_path) as f:
        f.extractall(output_dir)


def unzip(compressed_path, output_dir):
    with zipfile.ZipFile(compressed_path, 'r') as f:
        f.extractall(output_dir)


def ungzip(compressed_path, output_dir):
    filename = os.path.basename(compressed_path)
    assert filename.endswith('.gz')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, filename[:-3])
    with gzip.open(compressed_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def unbz2(compressed_path, output_dir):
    extract_filename = os.path.basename(compressed_path).replace('.bz2', '')
    extract_path = os.path.join(output_dir, extract_filename)
    with bz2.BZ2File(compressed_path, 'rb') as compressed_file, open(extract_path, 'wb') as extract_file:
        for data in tqdm(iter(lambda: compressed_file.read(1024 * 1024), b'')):
            extract_file.write(data)


def add_newline_at_end_of_file(file_path):
    with open(file_path, 'r') as f:
        last_character = f.readlines()[-1][-1]
    if last_character == '\n':
        return
    print(f'Adding newline at the end of {file_path}')
    with open(file_path, 'a') as f:
        f.write('\n')


def git_clone(url, output_dir, overwrite=True):
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    git.Repo.clone_from(url, output_dir)


def replace_lrb_rrb_file(filepath):
    tmp_filepath = filepath + '.tmp'
    with open(filepath, 'r') as input_file, open(tmp_filepath, 'w') as output_file:
        for line in input_file:
            output_file.write(line.replace('-lrb-', '(').replace('-rrb-', ')'))
    os.rename(tmp_filepath, filepath)
