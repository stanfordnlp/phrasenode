#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Example usage:
./server.py data/experiments/0_encoding-baseline/config.txt -m data/experiments/0_encoding-baseline/checkpoints/10000.checkpoint/model
"""

import argparse
import datetime
import socket
from os.path import join, exists

import bottle, json
from bottle import post, request, run, response
# Increase size to 10MB
bottle.BaseRequest.MEMFILE_MAX = 10 * 1024 * 1024

from gtd.io import save_stdout, IntegerDirectories
from gtd.log import set_log_level
from gtd.utils import Config

from phrasenode import data
from phrasenode.eval_run import PhraseNodeEvalRun


# CONFIGS ARE MERGED IN THE FOLLOWING ORDER:
# 1. configs in args.config_paths, from left to right
# 2. config_strings

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port', type=int, default=6006)
parser.add_argument('-m', '--model-file', default='')
parser.add_argument('-s', '--config-strings', action='append', default=[])
parser.add_argument('config_paths', nargs='+')
args = parser.parse_args()

set_log_level('WARNING')

# create run
config_strings = []
config_paths = args.config_paths
if len(config_paths) == 1 and config_paths[0].isdigit():
    # Get configs from a run
    run_dirname = IntegerDirectories(data.workspace.experiments)[int(config_paths[0])]
    with open(join(run_dirname, 'config.txt')) as fin:
        config_strings.append(fin.read())
else:
    # Merge strings to allow object overwites
    run_dirname = None
    for filename in config_paths:
        with open(filename) as fin:
            config_strings.append(fin.read())

for config_string in args.config_strings:
    config_strings.append(config_string)
config = Config.from_str('\n'.join(config_strings))
eval_run = PhraseNodeEvalRun(config)

# load model
if args.model_file and exists(args.model_file):
    # Load model from a file
    eval_run.load_model(args.model_file)
elif run_dirname is not None and args.model_file.isdigit():
    # Load a specific checkpoint
    model_file = join(run_dirname, 'checkpoints', args.model_file + '.checkpoint', 'model')
    eval_run.load_model(model_file)
else:
    raise ValueError('Cannot load model from "{}"'.format(args.model_file))


################################################
# SERVER CODE

@post('/pred')
def eval():
    q = request.forms
    phrase = q.query
    print('[{}] Received {}'.format(datetime.datetime.now().time(), phrase))
    info = json.loads(q.info)
    info = json.dumps(info, ensure_ascii=True)
    answer = eval_run.eval(phrase, info)
    return {'query': q.query, 'answer': answer['preds'][0]['xid']}


def start_server(port):
    # This will open a global port!
    print('[{}] Starting server'.format(datetime.datetime.now().time()))
    run(host='0.0.0.0', port=port)
    print('\nGood bye!')


start_server(args.port)
