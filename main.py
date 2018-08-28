#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import random
import socket
from os.path import join

import numpy as np
import torch

from gtd.io import save_stdout
from gtd.log import set_log_level
from gtd.utils import Config

from phrasenode.training_run import PhraseNodeTrainingRuns


# CONFIGS ARE MERGED IN THE FOLLOWING ORDER:
# 1. configs in args.config_paths, from left to right
# 2. config_strings

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--config_strings', action='append', default=[])
parser.add_argument('--check_commit', default='strict')
parser.add_argument('-p', '--profile', action='store_true')
parser.add_argument('-d', '--description', default='None.')
parser.add_argument('-n', '--name', default='unnamed')
parser.add_argument('-c', '--comment')
parser.add_argument('-r', '--seed', default=0)
parser.add_argument('config_paths', nargs='+')
args = parser.parse_args()

# Set the seeds
set_log_level('WARNING')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# create run
runs = PhraseNodeTrainingRuns(check_commit=(args.check_commit == 'strict'))

config_paths = args.config_paths
if len(config_paths) == 1 and config_paths[0].isdigit():
    # reload old run
    run = runs[int(config_paths[0])]
else:
    # Merge strings to allow object overwites
    config_strings = []
    for filename in config_paths:
        with open(filename) as fin:
            config_strings.append(fin.read())
    for config_string in args.config_strings:
        config_strings.append(config_string)
    config = Config.from_str('\n'.join(config_strings))
    run = runs.new(config, name=args.name)  # new run from config

    run.metadata['description'] = args.description
    run.metadata['name'] = args.name

run.metadata['host'] = socket.gethostname()

# start training
run.workspace.add_file('stdout', 'stdout.txt')
run.workspace.add_file('stderr', 'stderr.txt')
run.workspace.add_file('command', 'command.txt')
with open(run.workspace.command, 'a') as fout:
    print >> fout, sys.argv
if args.comment:
    run.workspace.add_file('comment', 'comment.txt')
    with open(run.workspace.comment, 'a') as fout:
        print >> fout, args.comment


if args.profile:
    from gtd.chrono import Profiling, Profiler
    profiler = Profiler.default()
    # To profile a class, do something like
    # import phrasenode.foo
    # profiler.add_module(phrasenode.foo)
    Profiling.start()

with save_stdout(run.workspace.root):
    try:
        run.load_latest_model()
        run.train()
    finally:
        run.close()
        if args.profile:
            Profiling.report()
