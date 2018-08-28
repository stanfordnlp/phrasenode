"""Embed nodes using information propagation"""
from collections import namedtuple, defaultdict
from itertools import izip
import random

import numpy as np
import torch
from torch import LongTensor as LT, FloatTensor as FT
import torch.nn as nn
import torch.nn.functional as F

from gtd.ml.torch.seq_batch import SequenceBatch
from gtd.ml.torch.token_embedder import TokenEmbedder
from gtd.ml.torch.utils import GPUVariable as V

from phrasenode.constants import UNK, EOS, HIDDEN, TAGS, GraphRels
from phrasenode.utterance_embedder import AverageUtteranceEmbedder, LSTMUtteranceEmbedder, AttentionUtteranceEmbedder
from phrasenode.utils import word_tokenize
from phrasenode.vocab import GloveEmbeddings, RandomEmbeddings, read_frequency_vocab


################################################
# Base Embedder

class ProppyBaseEmbedder(nn.Module):

    def __init__(self, dim, utterance_embedder, recursive_texts,
            attr_embed_dim, max_attr_tokens, min_id_freq, min_class_freq, dropout):
        """
        Args:
            dim (int): Target embedding dimension
            utterance_embedder (UtteranceEmbedder)
            recursive_texts (bool): For node text, whether to recursively combine the
                texts of the descendants
            attr_embed_dim (int): Size of each attribute embedding
            max_attr_tokens (int): Limit the number of attribute tokens to embed
            min_id_freq (int): Minimum token frequency of tokens in id vocab
            min_class_freq (int): Minimum token frequency of tokens in class vocab
            dropout (float): Dropout rate
        """
        super(ProppyBaseEmbedder, self).__init__()
        self._dim = dim
        # Text embedder
        self._utterance_embedder = utterance_embedder
        self._max_words = utterance_embedder.max_words
        self._recursive_texts = recursive_texts
        # Attribute embedders
        self._attr_embed_dim = attr_embed_dim
        tags = [UNK, EOS] + TAGS
        self._tag_embedder = \
                TokenEmbedder(RandomEmbeddings(tags, attr_embed_dim))
        ids = read_frequency_vocab('frequent-ids', min_id_freq)
        self._id_embedder = AverageUtteranceEmbedder(
                TokenEmbedder(RandomEmbeddings(ids, attr_embed_dim)), max_attr_tokens)
        classes = read_frequency_vocab('frequent-classes', min_class_freq)
        self._classes_embedder = AverageUtteranceEmbedder(
                TokenEmbedder(RandomEmbeddings(classes, attr_embed_dim)), max_attr_tokens)
        coords_dim = 3
        # Combine
        input_dim = (self._utterance_embedder.embed_dim
                + 3 * attr_embed_dim + coords_dim)
        self.dropout = nn.Dropout(dropout)
        #self.fc = nn.Linear(self._utterance_embedder.embed_dim, dim)
        self.fc = nn.Linear(input_dim, dim)

    @property
    def embed_dim(self):
        return self._dim

    @property
    def token_embedder(self):
        return self._utterance_embedder.token_embedder

    @property
    def utterance_embedder(self):
        return self._utterance_embedder

    def forward(self, nodes):
        """Embeds a batch of Nodes.

        Args:
            nodes (list[Node])
        Returns:
            embeddings (Tensor): num_nodes x embed_dim
        """
        texts = []
        for node in nodes:
            if self._recursive_texts:
                text = ' '.join(node.all_texts(max_words=self._max_words))
            else:
                text = node.text or ''
            texts.append(word_tokenize(text.lower()))
        text_embeddings = self._utterance_embedder(texts)

        ## num_nodes x attr_embed_dim
        tag_embeddings = self._tag_embedder.embed_tokens(
                [node.tag for node in nodes])
        # num_nodes x attr_embed_dim
        id_embeddings = self._id_embedder(
                [word_tokenize(node.id_) for node in nodes])
        # num_nodes x attr_embed_dim
        class_embeddings = self._classes_embedder(
                [word_tokenize(' '.join(node.classes)) for node in nodes])
        # num_nodes x 3
        coords = V(FT(
            [[elem.x_ratio, elem.y_ratio, float(elem.visible)]
                for elem in nodes]))

        # num_nodes x dom_embed_dim
        dom_embeddings = torch.cat(
                (text_embeddings, tag_embeddings, id_embeddings,
                    class_embeddings, coords), dim=1)
        #dom_embeddings = text_embeddings
        return self.fc(dom_embeddings)
        #return F.relu(self.fc(self.dropout(dom_embeddings)))
        #return F.sigmoid(self.fc(dom_embeddings))


################################################
# Information propagation

class ProppyEmbedder(nn.Module):

    def __init__(self, dim, base_embedder, iterations,
            neighbor_rels, max_neighbors, aggregator):
        """Initialize.

        Args:
            dim (int): Dimension of the base and final embedding.
            base_embedder (ProppyBaseEmbedder)
            iterations (int): Number of times to propagate information
            neighbor_rels (list[str]): Node A is a neighbor of Node B
                if there is a relation in neighbor_rels from A to B
            max_neighbors (int): Maximum number of neighbors for each relation
            aggregator (Aggregator)
        """
        super(ProppyEmbedder, self).__init__()
        self._dim = dim
        self._base_embedder = base_embedder
        self._iterations = iterations
        # relation string -> index
        self._neighbor_rels = {x: i for (i, x) in enumerate(sorted(set(neighbor_rels)))}
        self._max_neighbors = max_neighbors
        assert all(x in GraphRels.LOOKUP for x in self._neighbor_rels)
        self._aggregator = aggregator

    @property
    def embed_dim(self):
        return self._dim

    @property
    def token_embedder(self):
        return self._base_embedder.token_embedder

    @property
    def utterance_embedder(self):
        return self._base_embedder.utterance_embedder

    def forward(self, nodes, mask=None):
        """Embeds a batch of Nodes.

        Args:
            nodes (list[Node])
            mask (Tensor): 1D of length num_nodes
        Returns:
            embeddings (Tensor): num_nodes x embed_dim
        """
        # num_nodes x embed_dim
        embeds = self._base_embedder(nodes)
        batch_size = embeds.shape[0]
        if mask is not None:
            embeds = embeds.mul(mask.unsqueeze(1))
        for itr in xrange(self._iterations):
            neighbors, rels = self._get_neighbors(nodes[0].web_page)
            embeds = self._aggregator(embeds, neighbors, rels)
        return embeds

    def _get_neighbors(self, web_page):
        """Get indices of at most |max_neighbors| neighbors for each relation

        Args:
            web_page (WebPage)
        Returns:
            neighbors: SequenceBatch of shape num_nodes x ???
                containing the neighbor refs
                (??? is at most max_neighbors * len(neighbor_rels))
            rels: SequenceBatch of shape num_nodes x ???
                containing the relation indices
        """
        G = web_page.graph
        batch_neighbors = [[] for _ in xrange(len(web_page.nodes))]
        batch_rels = [[] for _ in xrange(len(web_page.nodes))]
        for src, tgts in G.nodes.iteritems():
            # Group by relation
            rel_to_tgts = defaultdict(list)
            for tgt, rels in tgts.iteritems():
                for rel in rels:
                    rel_to_tgts[rel].append(tgt)
            # Sample if needed
            for rel, index in self._neighbor_rels.iteritems():
                tgts = rel_to_tgts[rel]
                random.shuffle(tgts)
                if not tgts:
                    continue
                if len(tgts) > self._max_neighbors:
                    tgts = tgts[:self._max_neighbors]
                batch_neighbors[src].extend(tgts)
                batch_rels[src].extend([index] * len(tgts))
        # Create SequenceBatches
        max_len = max(len(x) for x in batch_neighbors)
        batch_mask = []
        for neighbors, rels in izip(batch_neighbors, batch_rels):
            assert len(neighbors) == len(rels)
            this_len = len(neighbors)
            batch_mask.append([1.] * this_len + [0.] * (max_len - this_len))
            neighbors.extend([0] * (max_len - this_len))
            rels.extend([0] * (max_len - this_len))
        return (SequenceBatch(V(LT(batch_neighbors)), V(FT(batch_mask))),
                SequenceBatch(V(LT(batch_rels)), V(FT(batch_mask))))


################################################
# Aggregator

class Aggregator(nn.Module):

    def forward(self, old_embeds, neighbors, rels):
        """
        Args:
            old_embeds (Tensor): batch_size x embed_dim
            neighbors (SequenceBatch): batch_size x num_neighbors
            rels (SequenceBatch): batch_size x num_neighbors
        Returns:
            new_embeds (Tensor): batch_size x embed_dim
        """
        raise NotImplementedError


class PoolMLPAggregator(Aggregator):

    def __init__(self, dim, dropout):
        super(PoolMLPAggregator, self).__init__()
        self._dropout = nn.Dropout(dropout)
        self._proj = nn.Linear(dim * 2, dim)

    def forward(self, old_embeds, neighbors, rels):
        batch_size = len(old_embeds)
        neighbor_embeds = torch.index_select(old_embeds, 0, neighbors.values.view(-1))
        neighbor_embeds = neighbor_embeds.view(batch_size, neighbors.values.shape[1], -1)
        neighbor_embeds = SequenceBatch(neighbor_embeds, neighbors.mask)
        pooled = SequenceBatch.reduce_max(neighbor_embeds)
        combined = torch.cat((old_embeds, pooled), dim=1)
        return F.relu(self._proj(self._dropout(combined)))


class MLPPoolAggregator(Aggregator):

    def __init__(self, dim, dropout):
        super(MLPPoolAggregator, self).__init__()
        self._dropout = nn.Dropout(dropout)
        self._proj = nn.Linear(dim, dim)

    def forward(self, old_embeds, neighbors, rels):
        batch_size = len(old_embeds)
        projected = F.relu(self._proj(self._dropout(old_embeds)))
        neighbor_embeds = torch.index_select(projected, 0, neighbors.values.view(-1))
        neighbor_embeds = neighbor_embeds.view(batch_size, neighbors.values.shape[1], -1)
        combined = torch.cat((projected.unsqueeze(1), neighbor_embeds), dim=1)
        mask = torch.cat((V(torch.ones(batch_size, 1)), neighbors.mask), dim=1)
        combined = SequenceBatch(combined, mask)
        return SequenceBatch.reduce_max(combined)


################################################
# Final model

def get_proppy_embedder(config):
    """Create a new ProppyEmbedder based on the config

    Args:
        config (Config): the root config
    Returns:
        ProppyEmbedder
    """
    cm = config.model
    cmu = cm.utterance_embedder
    # Token embedder
    glove_embeddings = GloveEmbeddings(cmu.vocab_size, cmu.glove_dim)
    token_embedder = TokenEmbedder(glove_embeddings, trainable=cmu.trainable)
    # Utterance embedder
    if cmu.type == 'average':
        utterance_embedder = AverageUtteranceEmbedder(token_embedder, cmu.max_words)
    elif cmu.type == 'lstm':
        utterance_embedder = LSTMUtteranceEmbedder(token_embedder, cmu.lstm_dim, cmu.max_words)
    elif cmu.type == 'attention_lstm':
        utterance_embedder = AttentionUtteranceEmbedder(token_embedder, cmu.lstm_dim, cmu.max_words)
    else:
        raise ValueError('Unknown UtteranceEmbedder type {}'.format(cmu.type))
    # Base node embedder
    cmb = cm.node_embedder.base_embedder
    base_embedder = ProppyBaseEmbedder(cm.dim,
            utterance_embedder, cmb.recursive_texts,
            cmb.attr_embed_dim, cmb.max_attr_tokens,
            cmb.min_id_freq, cmb.min_class_freq, cm.dropout)
    # Aggregator
    cmpr = cm.node_embedder.propagation
    if cmpr.iterations == 0:
        return base_embedder
    if cmpr.aggregator == 'pool_mlp':
        aggregator = PoolMLPAggregator(cm.dim, cm.dropout)
    elif cmpr.aggregator == 'mlp_pool':
        aggregator = MLPPoolAggregator(cm.dim, cm.dropout)
    else:
        raise ValueError('Unknown Aggregator {}'.format(cmpr.aggregator))
    # Information propagation
    full_embedder = ProppyEmbedder(cm.dim, base_embedder, cmpr.iterations,
            cmpr.neighbor_rels, cmpr.max_neighbors, aggregator)
    return full_embedder
