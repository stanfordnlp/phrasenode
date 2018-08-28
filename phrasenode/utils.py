import logging
import os
import re

import numpy as np

from phrasenode.constants import UNK


def create_experiment_dir(outdir):
    assert os.path.isdir(outdir)
    max_id = -1
    for name in os.listdir(outdir):
        match = re.match(r'^(\d+).*$', name)
        if match:
            max_id = max(max_id, int(match.group(1)))
    id_ = max_id + 1
    exp_dir = os.path.join(outdir, str(id_))
    os.mkdir(exp_dir)
    return exp_dir


################################################
# Tokenization

TOKENIZER = re.compile(r'[^\W_]+|[^\w\s-]', re.UNICODE | re.MULTILINE | re.DOTALL)

def word_tokenize(text):
    """Tokenize without keeping the mapping to the original string.

    Args:
        text (str or unicode)
    Return:
        list[unicode]
    """
    return TOKENIZER.findall(text)



TOKENIZER2 = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w]+", re.UNICODE | re.MULTILINE | re.DOTALL)


# courtesy https://stackoverflow.com/questions/6202549/word-tokenization-using-python-regular-expressions
def word_tokenize2(text):
    """Tokenize without keeping the mapping to the original string.
    Removes punctuation, keeps dashes, and splits on capital letters correctly.
    Returns tokenized words in lower case.
    E.g.
    Jeff's dog is un-American SomeTimes! BUT NOTAlways
    ['jeff's', 'dog', 'is', 'un', 'american', 'some', 'times', 'but', 'not', 'always']


    Args:
        text (str or unicode)
    Return:
        list[unicode]
    """
    return [s.lower() for s in TOKENIZER2.findall(text)]



################################################
# Geometry

def rect_area(r):
    """Return the area of a rectangle.

    Args:
        r: an object with attributes left, top, width, height
    Returns:
        float
    """
    return float(r.width) * float(r.height)


def rect_overlap(r1, r2):
    """Return the area of the intersection of two rectangles.

    Args:
        r1: an object with attributes left, top, width, height
        r2: an object with attributes left, top, width, height
    Returns:
        float
    """
    left = float(max(r1.left, r2.left))
    right = float(min(r1.left + r1.width, r2.left + r2.width))
    top = float(max(r1.top, r2.top))
    bottom = float(min(r1.top + r1.height, r2.top + r2.height))
    if left >= right or top >= bottom:
        return 0.
    return (right - left) * (bottom - top)


################################################
# Statistics

class Stats(object):
    
    def __init__(self):
        self.n = 0
        self.loss = 0.
        self.accuracy = 0.
        self.area_f1 = 0.
        self.oracle = 0.
        self.str_acc = 0.
        self.grad_norm = 0.

    def add(self, stats):
        """Add another Stats to this one."""
        self.n += stats.n
        self.loss += stats.loss
        self.accuracy += stats.accuracy
        self.area_f1 += stats.area_f1
        self.oracle += stats.oracle
        self.str_acc += stats.str_acc
        self.grad_norm = max(self.grad_norm, stats.grad_norm)

    def __repr__(self):
        n = max(1, self.n) * 1.
        return '(n={}, loss={}, accuracy={}, area_f1={}, oracle={}, str_acc={}, grad_norm={})'\
                .format(self.n, self.loss / n, self.accuracy / n, self.area_f1 / n,
                        self.oracle / n, self.str_acc / n, self.grad_norm)
    __str__ = __repr__

    def log(self, tb_logger, step, prefix='', ignore_grad_norm=False):
        """Log to TensorBoard."""
        n = float(self.n)
        tb_logger.log_value(prefix + 'loss', self.loss / n, step)
        tb_logger.log_value(prefix + 'accuracy', self.accuracy / n, step)
        tb_logger.log_value(prefix + 'area_f1', self.area_f1 / n, step)
        tb_logger.log_value(prefix + 'oracle', self.oracle / n, step)
        tb_logger.log_value(prefix + 'str_acc', self.str_acc / n, step)
        if not ignore_grad_norm:
            tb_logger.log_value(prefix + 'grad_norm', self.grad_norm, step)
