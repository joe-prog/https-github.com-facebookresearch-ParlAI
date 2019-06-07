#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.

import parlai.core.build_data as build_data
import os
import json
import gzip

VERSION = '2'

IN_DOMAIN_DATASETS = [
    'SQuAD',
    'NewsQA',
    'TriviaQA-web',
    'SearchQA',
    'HotpotQA',
    'NaturalQuestionsShort'
]

OUT_DOMAIN_DATASETS = [
    'TextbookQA',
    'RelationExtraction',
    'DROP',
    'DuoRC.ParaphraseRC',
    'RACE'
]

URL = 'https://s3.us-east-2.amazonaws.com/mrqa/release'

BioASQ_URL = 'http://participants-area.bioasq.org/MRQA2019/'

QUESTION_SEPARATOR = '[Q]'

OUTPUT_FORMAT = (
    'text:{text}\t'
    'detected_answers:{detected_answers}\t'
    'labels:{labels}'
)


def _handle_data_point(data_point):
    outputs = []
    # qas = A list of questions for the given context.

    for qas in data_point['qas']:

        # take the first detected_answer as label
        output = OUTPUT_FORMAT.format(
            text=f"{data_point['context']} {QUESTION_SEPARATOR} {qas['question']}",
            detected_answers=qas['detected_answers'],
            labels=qas['detected_answers'][0]['text'],
        )
        output += '\t\tepisode_done:True\n'
        outputs.append(output)
    return outputs


def make_parlai_format(outpath, dtype, data):
    print('building parlai:' + dtype)
    with open(os.path.join(outpath, dtype + '.txt'), 'w') as fout:
        for data_point in data:
            for output in _handle_data_point(data_point):
                fout.write(output)


def build(opt):
    dpath = os.path.join(opt['datapath'], 'MRQA')
    version = VERSION

    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            # An older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # merge all train data
        train_data = []

        # train
        for dataset in IN_DOMAIN_DATASETS:

            fname = f'{dataset}.jsonl.gz'
            url_train = f'{URL}/v{VERSION}/train/{fname}'

            # Download the data.
            build_data.download(url_train, dpath, f'train_{fname}')

            # digest data
            with gzip.open(os.path.join(dpath,  f'train_{fname}'), "r") as f:
                data = [json.loads(line) for line in f if line]
                # exclude the header
                train_data.extend( data[1:] )

        make_parlai_format(dpath, 'train', train_data)

        # dev
        for dataset in (IN_DOMAIN_DATASETS + OUT_DOMAIN_DATASETS):

            fname = f'{dataset}.jsonl.gz'
            url_dev = f'{URL}/v{VERSION}/dev/{fname}'

            # Download the data.
            build_data.download(url_dev, dpath, f'dev_{fname}')

            # digest data
            with gzip.open(os.path.join(dpath, f'dev_{fname}'), "r") as f:
                dev_data = [json.loads(line) for line in f]
                make_parlai_format(dpath, f'dev_{dataset}', dev_data[1:])

        # BioASQ dataset
        dataset = 'BioASQ'
        fname = f'{dataset}.jsonl.gz'
        build_data.download(BioASQ_URL, dpath, f'dev_{fname}')
        with gzip.open(os.path.join(dpath, f'dev_{fname}'), "r") as f:
            dev_data = [json.loads(line) for line in f]
            make_parlai_format(dpath, f'dev_{dataset}', dev_data[1:])

        # Mark the data as built.
        build_data.mark_done(dpath, version_string=version)
