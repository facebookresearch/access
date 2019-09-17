# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from collections import defaultdict
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
import time

from fairseq import options
from fairseq_cli import preprocess, train, generate

from access.resources.paths import get_dataset_dir, EXP_DIR
from access.utils.helpers import (log_stdout, lock_directory, create_directory_or_skip, yield_lines,
                                  write_lines)


def get_fairseq_exp_dir(job_id=None):
    if job_id is not None:
        dir_name = f'slurmjob_{job_id}'
    else:
        dir_name = f'local_{int(time.time() * 1000)}'
    return Path(EXP_DIR) / f'fairseq' / dir_name


def fairseq_preprocess(dataset):
    dataset_dir = get_dataset_dir(dataset)
    with lock_directory(dataset_dir):
        preprocessed_dir = dataset_dir / 'fairseq_preprocessed'
        with create_directory_or_skip(preprocessed_dir):
            preprocessing_parser = options.get_preprocessing_parser()
            preprocess_args = preprocessing_parser.parse_args([
                '--source-lang',
                'complex',
                '--target-lang',
                'simple',
                '--trainpref',
                os.path.join(dataset_dir, f'{dataset}.train'),
                '--validpref',
                os.path.join(dataset_dir, f'{dataset}.valid'),
                '--testpref',
                os.path.join(dataset_dir, f'{dataset}.test'),
                '--destdir',
                str(preprocessed_dir),
                '--output-format',
                'raw',
            ])
            preprocess.main(preprocess_args)
        return preprocessed_dir


def fairseq_train(
        preprocessed_dir,
        exp_dir,
        ngpus=None,
        max_tokens=2000,
        arch='fconv_iwslt_de_en',
        pretrained_emb_path=None,
        embeddings_dim=None,
        # Transformer (decoder is the same as encoder for now)
        encoder_embed_dim=512,
        encoder_layers=6,
        encoder_attention_heads=8,
        # encoder_decoder_dim_ratio=1,
        # share_embeddings=True,
        max_epoch=50,
        warmup_updates=None,
        lr=0.1,
        min_lr=1e-9,
        dropout=0.2,
        label_smoothing=0.1,
        lr_scheduler='fixed',
        weight_decay=0.0001,
        criterion='label_smoothed_cross_entropy',
        optimizer='nag',
        validations_before_sari_early_stopping=10,
        fp16=False):
    exp_dir = Path(exp_dir)
    with log_stdout(exp_dir / 'fairseq_train.stdout'):
        preprocessed_dir = Path(preprocessed_dir)
        exp_dir.mkdir(exist_ok=True, parents=True)
        # Copy dictionaries to exp_dir for generation
        shutil.copy(preprocessed_dir / 'dict.complex.txt', exp_dir)
        shutil.copy(preprocessed_dir / 'dict.simple.txt', exp_dir)
        train_parser = options.get_training_parser()
        # if share_embeddings:
        #     assert encoder_decoder_dim_ratio == 1
        args = [
            '--task',
            'translation',
            preprocessed_dir,
            '--raw-text',
            '--source-lang',
            'complex',
            '--target-lang',
            'simple',
            '--save-dir',
            os.path.join(exp_dir, 'checkpoints'),
            '--clip-norm',
            0.1,
            '--criterion',
            criterion,
            '--no-epoch-checkpoints',
            '--save-interval-updates',
            5000,  # Validate every n updates
            '--validations-before-sari-early-stopping',
            validations_before_sari_early_stopping,
            '--arch',
            arch,

            # '--decoder-out-embed-dim', int(embeddings_dim * encoder_decoder_dim_ratio),  # Output dim of decoder
            '--max-tokens',
            max_tokens,
            '--max-epoch',
            max_epoch,
            '--lr-scheduler',
            lr_scheduler,
            '--dropout',
            dropout,
            '--lr',
            lr,
            '--lr-shrink',
            0.5,  # For reduce lr on plateau scheduler
            '--min-lr',
            min_lr,
            '--weight-decay',
            weight_decay,
            '--optimizer',
            optimizer,
            '--label-smoothing',
            label_smoothing,
            '--seed',
            random.randint(1, 1000),
            # '--force-anneal', '200',
            # '--distributed-world-size', '1',
        ]
        if arch == 'transformer':
            args.extend([
                '--encoder-embed-dim',
                encoder_embed_dim,
                '--encoder-ffn-embed-dim',
                4 * encoder_embed_dim,
                '--encoder-layers',
                encoder_layers,
                '--encoder-attention-heads',
                encoder_attention_heads,
                '--decoder-layers',
                encoder_layers,
                '--decoder-attention-heads',
                encoder_attention_heads,
            ])
        if pretrained_emb_path is not None:
            args.extend(['--encoder-embed-path', pretrained_emb_path if pretrained_emb_path is not None else ''])
            args.extend(['--decoder-embed-path', pretrained_emb_path if pretrained_emb_path is not None else ''])
        if embeddings_dim is not None:
            args.extend(['--encoder-embed-dim', embeddings_dim])  # Input and output dim of encoder
            args.extend(['--decoder-embed-dim', embeddings_dim])  # Input dim of decoder
        if ngpus is not None:
            args.extend(['--distributed-world-size', ngpus])
        # if share_embeddings:
        #     args.append('--share-input-output-embed')
        if fp16:
            args.append('--fp16')
        if warmup_updates is not None:
            args.extend(['--warmup-updates', warmup_updates])
        args = [str(arg) for arg in args]
        train_args = options.parse_args_and_arch(train_parser, args)
        train.main(train_args)


def _fairseq_generate(complex_filepath,
                      output_pred_filepath,
                      checkpoint_paths,
                      complex_dictionary_path,
                      simple_dictionary_path,
                      beam=5,
                      hypothesis_num=1,
                      lenpen=1.,
                      diverse_beam_groups=None,
                      diverse_beam_strength=0.5,
                      sampling=False,
                      batch_size=128):
    # exp_dir must contain checkpoints/checkpoint_best.pt, and dict.{complex,simple}.txt
    # First copy input complex file to exp_dir and create dummy simple file
    tmp_dir = Path(tempfile.mkdtemp())
    new_complex_filepath = tmp_dir / 'tmp.complex-simple.complex'
    dummy_simple_filepath = tmp_dir / 'tmp.complex-simple.simple'
    shutil.copy(complex_filepath, new_complex_filepath)
    shutil.copy(complex_filepath, dummy_simple_filepath)
    shutil.copy(complex_dictionary_path, tmp_dir / 'dict.complex.txt')
    shutil.copy(simple_dictionary_path, tmp_dir / 'dict.simple.txt')
    generate_parser = options.get_generation_parser()
    args = [
        tmp_dir,
        '--path',
        ':'.join([str(path) for path in checkpoint_paths]),
        '--beam',
        beam,
        '--nbest',
        hypothesis_num,
        '--lenpen',
        lenpen,
        '--diverse-beam-groups',
        diverse_beam_groups if diverse_beam_groups is not None else -1,
        '--diverse-beam-strength',
        diverse_beam_strength,
        '--batch-size',
        batch_size,
        '--raw-text',
        '--print-alignment',
        '--gen-subset',
        'tmp',
        # We don't want to reload pretrained embeddings
        '--model-overrides',
        {
            'encoder_embed_path': None,
            'decoder_embed_path': None
        },
    ]
    if sampling:
        args.extend([
            '--sampling',
            '--sampling-topk',
            10,
        ])
    args = [str(arg) for arg in args]
    generate_args = options.parse_args_and_arch(generate_parser, args)
    out_filepath = tmp_dir / 'generation.out'
    with log_stdout(out_filepath, mute_stdout=True):
        # evaluate model in batch mode
        generate.main(generate_args)
    # Retrieve translations

    def parse_all_hypotheses(out_filepath):
        hypotheses_dict = defaultdict(list)
        for line in yield_lines(out_filepath):
            match = re.match(r'^H-(\d+)\t-?\d+\.\d+\t(.*)$', line)
            if match:
                sample_id, hypothesis = match.groups()
                hypotheses_dict[int(sample_id)].append(hypothesis)
        # Sort in original order
        return [hypotheses_dict[i] for i in range(len(hypotheses_dict))]

    all_hypotheses = parse_all_hypotheses(out_filepath)
    predictions = [hypotheses[hypothesis_num - 1] for hypotheses in all_hypotheses]
    write_lines(predictions, output_pred_filepath)
    os.remove(dummy_simple_filepath)
    os.remove(new_complex_filepath)


def fairseq_generate(complex_filepath,
                     output_pred_filepath,
                     exp_dir,
                     beam=1,
                     hypothesis_num=1,
                     lenpen=1.,
                     diverse_beam_groups=None,
                     diverse_beam_strength=0.5,
                     sampling=False,
                     batch_size=128):
    exp_dir = Path(exp_dir)
    checkpoint_path = exp_dir / 'checkpoints/checkpoint_best.pt'
    assert checkpoint_path.exists(), f'Generation failed, no checkpoint at {checkpoint_path}'
    complex_dictionary_path = exp_dir / 'dict.complex.txt'
    simple_dictionary_path = exp_dir / 'dict.simple.txt'
    _fairseq_generate(complex_filepath,
                      output_pred_filepath, [checkpoint_path],
                      complex_dictionary_path=complex_dictionary_path,
                      simple_dictionary_path=simple_dictionary_path,
                      beam=beam,
                      hypothesis_num=hypothesis_num,
                      lenpen=lenpen,
                      diverse_beam_groups=diverse_beam_groups,
                      diverse_beam_strength=diverse_beam_strength,
                      sampling=sampling,
                      batch_size=batch_size)
