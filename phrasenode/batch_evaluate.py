#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run evaluation on an external prediction JSONL file.

Input format: Each line is a JSON with at least the following attributes:
- exampleId (string)
- predictions (array):
    - Each item is an object with key: xid (int or string)

Output format: Copy of the dataset but with the following additional attributes:
- predictions (array):
    - Each item is an object with keys: xid (int), ref (int), match (bool),
        prec (float), rec (float), f1 (float)
    - Also have all additional keys provided in the input's predictions object
"""

import sys, os, shutil, re, argparse, json
from codecs import open
from itertools import izip
from collections import defaultdict, Counter

from tqdm import tqdm

from phrasenode import data
from phrasenode.dataset import PhraseNodeStorage
from phrasenode.utils import Stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--good-xids',
            help='Specify the path to good-xids to filter the candidates')
    parser.add_argument('set_name',
            help='Dataset set name (e.g., v3)')
    parser.add_argument('infile',
            help='Path to the prediction JSONL file')
    args = parser.parse_args()

    # Read the input file
    inputs = {}
    with open(args.infile) as fin:
        for line in fin:
            line = json.loads(line)
            assert 'exampleId' in line
            assert 'predictions' in line
            inputs[line['exampleId']] = line['predictions']
    print >> sys.stderr, 'Read {} input lines from {}'.format(
            len(inputs), args.infile)

    # Read good-xids
    if args.good_xids:
        good_xids = {}
        with open(args.good_xids) as fin:
            for line in fin:
                line = json.loads(line)
                good_xids[line['version'], line['webpage']] = line['xids']
        print >> sys.stderr, 'Read {} good xids entries'.format(
                len(good_xids))

    storage = PhraseNodeStorage(data.workspace.phrase_node)
    all_examples = storage.load_examples(args.set_name)
    stats = Stats()

    for web_page_code, examples in tqdm(all_examples.items()):
        web_page = storage.get_web_page(web_page_code, check=False)
        for example in examples:
            stats.n += 1
            target_ref = web_page.xid_to_ref.get(example.target_xid)
            answer = example.clone_metadata()
            answer['predictions'] = []
            # Go through the predictions
            read_top_prediction = False
            for prediction in inputs.get(example.example_id, []):
                prediction = dict(prediction)
                if 'xid' not in prediction:
                    continue
                pred_xid = int(prediction['xid'])
                if args.good_xids and pred_xid not in good_xids[web_page_code]:
                    continue
                prediction['match'] = (example.target_xid == pred_xid)
                pred_ref = web_page.xid_to_ref.get(pred_xid)
                if pred_ref is None:
                    prediction['prec'] = prediction['rec'] = prediction['f1'] = 0.
                else:
                    prediction['prec'], prediction['rec'], prediction['f1'] = \
                            web_page.overlap_eval(target_ref, pred_ref)
                answer['predictions'].append(prediction)
                # Augment stats
                if not read_top_prediction:
                    stats.accuracy += prediction['match']
                    stats.area_f1 += prediction['f1']
                    read_top_prediction = True
            print json.dumps(answer)

    print >> sys.stderr, stats

if __name__ == '__main__':
    main()
