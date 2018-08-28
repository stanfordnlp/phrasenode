"""Similar to PhraseNodeTrainingRun. The differences are:
- Does not have an experiment directory
- Does not have an optimizer
- Supports querying one example on the fly.
"""
import gzip
import json
import logging
import os
from os.path import dirname, realpath, join

import torch
from gtd.ml.torch.utils import try_gpu

from phrasenode import data
from phrasenode.dataset import PhraseNodeStorage
from phrasenode.model import create_model
from phrasenode.utils import Stats


class PhraseNodeEvalRun(object):
    """Similar to TrainingRun but does not include an optimizer."""

    def __init__(self, config):
        self.config = config
        self._create_model()

    def _create_model(self):
        config = self.config
        self.model = create_model(config)
        self.model = try_gpu(self.model)

    def load_model(self, model_file):
        print 'Loading from file {}'.format(model_file)
        try:
            state_dict = torch.load(model_file)
        except:
            logging.warning('Cannot load into GPU. Try CPU.')
            state_dict = torch.load(model_file, map_location='cpu')
        self.model.load_state_dict(state_dict)
        print 'Model loaded.'

    def eval(self, phrase, info):
        # write example and write info
        example = {"exampleId": "eval", "phrase": phrase, "version": "eval", "webpage": "eval", "xid": 1}
        example_path = join(os.environ['WEBREP_DATA'], 'phrase-node-dataset', 'data', 'eval.jsonl')
        info_path = join(os.environ['WEBREP_DATA'], 'phrase-node-dataset', 'infos', 'eval', 'info-eval.gz')

        with open(example_path, 'wb') as f:
            json.dump(example, f)
        with gzip.open(info_path, 'wb') as f:
            # info is already JSON-serialized
            f.write(info)

        out = []
        storage = PhraseNodeStorage(data.workspace.phrase_node)
        loaded_examples = storage.load_examples('eval')
        for _, examples in loaded_examples.items():
            result = self._process_one(examples)
            out.append(result)
        return out[0]

    def _process_one(self, examples):
        """Process example from the same web page.

        Args:
            example [PhraseNodeExample]
        Returns:
            Stats
        """
        stats = Stats()
        web_page = examples[0].get_web_page()
        if not web_page:
            return stats

        logits, losses, predictions = self.model(web_page, examples)
        # loss
        averaged_loss = torch.sum(losses) / len(examples)

        stats.n = len(examples)
        stats.loss = float(torch.sum(losses))
        stats.logits = logits
        stats.losses = losses
        stats.predictions = predictions
        # evaluate
        for i, example in enumerate(examples):
            # Top prediction
            pred_ref = predictions.data[i][0]
            pred_node = web_page[pred_ref]
            pred_xid = pred_node.xid
            match = (example.target_xid == pred_xid)

            target_ref = web_page.xid_to_ref.get(example.target_xid)
            target_node = web_page[target_ref] if target_ref is not None else None
            prec, rec, f1 = web_page.overlap_eval(target_ref, pred_ref)

            str_acc = self._check_str(pred_node, target_node)

            # Oracle
            oracle = bool(target_ref is not None and logits.data[i, target_ref] > -99999)

            stats.accuracy += float(match)
            stats.area_f1 += f1
            stats.str_acc += float(str_acc)
            stats.oracle += float(oracle)

            metadata = example.clone_metadata()
            metadata['oracle'] = oracle
            metadata['predictions'] = []
            pred_xids = set()
            for pred_ref in predictions.data[i]:
                pred_node = web_page[pred_ref]
                pred_xid = pred_node.xid
                pred_xids.add(pred_xid)
                match = (example.target_xid == pred_xid)
                prec, rec, f1 = web_page.overlap_eval(target_ref, pred_ref)
                str_acc = self._check_str(pred_node, target_node)
                metadata['predictions'].append({
                    'xid': pred_xid, 'score': float(logits.data[i, pred_ref]),
                    'match': match,
                    'prec': prec, 'rec': rec, 'f1': f1,
                    'str_acc': str_acc,
                    })
            if example.target_xid not in pred_xids and target_ref is not None:
                prec, rec, f1 = web_page.overlap_eval(target_ref, target_ref)
                str_acc = self._check_str(target_node, target_node)
                metadata['predictions'].append({
                    'xid': example.target_xid, 'score': float(logits.data[i, target_ref]),
                    'match': True,
                    'prec': prec, 'rec': rec, 'f1': f1,
                    'str_acc': str_acc
                    })
        return {'oracle': stats.oracle,
            'str_acc': stats.str_acc,
            'loss': stats.loss,
            'phrase': metadata['phrase'],
            'preds': metadata['predictions']}

    def _check_str(self, pred_node, target_node):
        """Check if the strings of the two nodes are identical."""
        if pred_node is None or target_node is None:
            return False
        pred_text = pred_node.all_texts(max_words=10)
        target_node = target_node.all_texts(max_words=10)
        return ' '.join(pred_text).lower() == ' '.join(target_node).lower()
