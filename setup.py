# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from setuptools import setup, find_packages


with open('README.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f]

setup(
    name='access',
    version='0.2',
    description='Controllable Sentence Simplification',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Louis Martin <louismartincs@gmail.com>',
    url='https://github.com/facebookreasearch/access',
    packages=find_packages(exclude=['resources']),
    install_requires=requirements,
)
