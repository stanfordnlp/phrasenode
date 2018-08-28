"""Main entry point for training a model.
- Parse the config and create objects.
- Load the necessary resources.
- Train the model, with periodic evaluation.
"""
import gzip
import json
import logging
import os
from os.path import dirname, realpath, join
import random

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm

from gtd.ml.torch.training_run import TorchTrainingRun
from gtd.ml.torch.utils import try_gpu, random_seed
from gtd.ml.training_run import TrainingRuns
from gtd.utils import as_batches

from phrasenode import data
from phrasenode.dataset import PhraseNodeStorage
from phrasenode.model import create_model
from phrasenode.utils import Stats
from phrasenode.webpage import WebPage, check_web_page


class PhraseNodeTrainingRuns(TrainingRuns):
    def __init__(self, check_commit=True):
        data_dir = data.workspace.experiments
        src_dir = dirname(realpath(join(__file__, '..')))
        super(PhraseNodeTrainingRuns, self).__init__(
            data_dir, src_dir, PhraseNodeTrainingRun, check_commit=check_commit)


class PhraseNodeTrainingRun(TorchTrainingRun):
    """Encapsulates the elements of a training run."""

    def __init__(self, config, save_dir):
        super(PhraseNodeTrainingRun, self).__init__(config, save_dir)
        self._verify_config()
        self._init_log()
        self._create_model()
        self._read_dataset()

    def _verify_config(self):
        """Check the sanity of the config values."""
        config = self.config

    def _init_log(self):
        self.workspace.add_dir('logs', 'logs')

    def _read_dataset(self):
        config = self.config
        storage = PhraseNodeStorage(data.workspace.phrase_node)
        self.train_data = storage.load_examples(config.data.dataset + '.train')
        self.dev_data = storage.load_examples(config.data.dataset + '.dev')
        self.test_data = storage.load_examples(config.data.dataset + '.test')

    def close(self):
        # Close the log file
        try:
            self.logfile.close()
        except:
            pass

    ################################################

    def _create_model(self):
        config = self.config
        self.model = create_model(config)
        self.model = try_gpu(self.model)
        self.optimizer = optim.Adam(self.model.parameters(),
                lr=self.config.train.learning_rate,
                weight_decay=self.config.train.l2_reg)
        self.gradient_clip = config.train.gradient_clip

    def load_model(self, train_steps):
        print 'Loading from checkpoint {}'.format(train_steps)
        self.train_state = self.checkpoints.load(
                train_steps, self.model, self.optimizer)

    def load_latest_model(self):
        print 'Loading the latest checkpoint'
        self.train_state = self.checkpoints.load_latest(
                self.model, self.optimizer)

    def save_model(self):
        """Save the current model to a checkpoint."""
        print 'Saving model to {}.checkpoint'.format(self.train_state.train_steps)
        self.checkpoints.save(self.train_state)

    def prune_old_models(self):
        """Prune old models to save space."""
        clc = self.config.log.checkpoints
        checkpoint_numbers = self.checkpoints.checkpoint_numbers
        to_keep = (set(checkpoint_numbers[-clc.keep_last:]) |
                set(x for x in checkpoint_numbers if x % clc.keep_every == 0))
        print 'Keeping checkpoints {}'.format(sorted(to_keep))
        to_prune = [x for x in checkpoint_numbers if x not in to_keep]
        for x in to_prune:
            self.checkpoints.delete(x)

    ################################################

    def train(self):
        config = self.config

        # Dummy values
        train_iterator = iter([])
        self.logfile = open('/dev/null')
        train_stats = Stats()

        # Training loop
        min_step = self.train_state.train_steps + 1
        max_step = config.timing.max_control_steps
        for step in tqdm(xrange(min_step, max_step + 1), desc='Training'):
            self.train_state.increment_train_steps()
            assert step == self.train_state.train_steps

            # Grab a web page from the training dataset
            try:
                web_page_code, examples = train_iterator.next()
            except StopIteration:
                train_iterator = iter(self._get_data_group_list(self.train_data, shuffle=True))
                self.logfile.close()
                self.logfile = gzip.open(
                        join(self.workspace.logs, 'eval-pn-train.{}.gz'.format(step)), 'w')
                web_page_code, examples = train_iterator.next()

            # Train
            ex_stats = self._process_examples(
                    web_page_code, examples, train=True, logfile=self.logfile)
            train_stats.add(ex_stats)

            # Save the model
            if step % config.timing.save_freq == 0:
                self.save_model()
                self.prune_old_models()

            # Log the results
            if step % config.timing.log_freq == 0:
                # TODO: Do we really want to average over the entire history?
                train_stats.log(self.tb_logger, step, prefix='pn_train_')
                train_stats = Stats()

            # Evaluate
            if step % config.timing.dev_freq == 0:
                filename = join(self.workspace.logs, 'eval-pn-dev.{}.gz'.format(step))
                with gzip.open(filename, 'w') as dev_logfile:
                    dev_stats = Stats()
                    for web_page_code, examples in tqdm(
                            self._get_data_group_list(self.dev_data),
                            desc='Evaluate on dev set'):
                        ex_stats = self._process_examples(
                                web_page_code, examples, train=False, logfile=dev_logfile)
                        dev_stats.add(ex_stats)
                    print 'DEV @ {}: {}'.format(step, dev_stats)
                    dev_stats.log(self.tb_logger, step, 'pn_dev_', ignore_grad_norm=True)

            if step % config.timing.test_freq == 0:
                filename = join(self.workspace.logs, 'eval-pn-test.{}.gz'.format(step))
                with gzip.open(filename, 'w') as test_logfile:
                    test_stats = Stats()
                    for web_page_code, examples in tqdm(
                            self._get_data_group_list(self.test_data),
                            desc='Evaluate on test set'):
                        ex_stats = self._process_examples(
                                web_page_code, examples, train=False, logfile=test_logfile)
                        test_stats.add(ex_stats)
                    print 'TEST @ {}: {}'.format(step, test_stats)
                    test_stats.log(self.tb_logger, step, 'pn_test_', ignore_grad_norm=True)

    def _get_data_group_list(self, data_dict, shuffle=False):
        """Return a list of (web_page_code, examples)
        - web_page_code is tuple(str, str)
        - examples is list[PhraseNodeExample]
        """
        groups = data_dict.items()
        if shuffle:
            random.shuffle(groups)
        else:
            groups.sort()
        return groups

    def _process_examples(self, web_page_code, examples, train=False, logfile=None):
        """Process examples from the same web page.

        Args:
            web_page_code (tuple(str,str))
            examples (list[PhraseNodeExample])
            train (bool): Whether to do gradient update
            logfile (file): If provided, dump prediction to this file
        Returns:
            Stats
        """
        stats = Stats()
        web_page = examples[0].get_web_page()
        if not web_page:
            logging.warning('Cannot get web page from %s', web_page_code)
            return stats
        if train:
            self.optimizer.zero_grad()
            self.model.train()
        else:
            self.model.eval()
        logits, losses, predictions = self.model(web_page, examples)
        # loss
        averaged_loss = torch.sum(losses) / len(examples)
        stats.n = len(examples)
        stats.loss = float(torch.sum(losses))
        # evaluate
        for i, example in enumerate(examples):
            # Top prediction
            pred_ref = predictions.data[i][0]
            pred_node = web_page[pred_ref]
            pred_xid = pred_node.xid
            match = (example.target_xid == pred_xid)
            stats.accuracy += float(match)
            target_ref = web_page.xid_to_ref.get(example.target_xid)
            target_node = web_page[target_ref] if target_ref is not None else None
            prec, rec, f1 = web_page.overlap_eval(target_ref, pred_ref)
            stats.area_f1 += f1
            str_acc = self._check_str(pred_node, target_node)
            stats.str_acc += float(str_acc)
            # Oracle
            oracle = bool(target_ref is not None and logits.data[i, target_ref] > -99999)
            stats.oracle += float(oracle)
            # Log to log file
            if logfile:
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
                        'str_acc': str_acc,
                        })
                print >> logfile, json.dumps(metadata)
        # gradient
        if train and averaged_loss.requires_grad:
            averaged_loss.backward()
            stats.grad_norm = clip_grad_norm(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
        return stats

    def _check_str(self, pred_node, target_node):
        """Check if the strings of the two nodes are identical."""
        if pred_node is None or target_node is None:
            return False
        pred_text = pred_node.all_texts(max_words=10)
        target_node = target_node.all_texts(max_words=10)
        return ' '.join(pred_text).lower() == ' '.join(target_node).lower()
