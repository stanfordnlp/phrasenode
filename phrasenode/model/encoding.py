"""p(node|text) prediction model based on dot product of embeddings."""
import logging
import math, random
from collections import namedtuple, defaultdict
from itertools import izip

import numpy as np

import torch
import torch.optim as optim
from torch import LongTensor as LT, FloatTensor as FT
import torch.nn as nn
import torch.nn.functional as F

from gtd.ml.torch.seq_batch import SequenceBatch
from gtd.ml.torch.token_embedder import TokenEmbedder
from gtd.ml.torch.utils import GPUVariable as V
from gtd.ml.torch.utils import try_gpu

from phrasenode.constants import UNK, EOS, HIDDEN, TAGS, GraphRels
from phrasenode.node_filter import get_node_filter
from phrasenode.utterance_embedder import AverageUtteranceEmbedder, LSTMUtteranceEmbedder
from phrasenode.utils import word_tokenize
from phrasenode.vocab import GloveEmbeddings, RandomEmbeddings, read_frequency_vocab


class EncodingModel(nn.Module):

    def __init__(self, phrase_embedder, node_embedder, node_filter, top_k=5, project=False, dropout=0.3,
        use_neighbors=False, neighbor_rels=['above','below','left','right'], max_neighbors=1):
        """
        Args:
            phrase_embedder (UtteranceEmbedder)
            node_embedder (NodeEmbedder)
            node_filter (callable[(WebPage, web_page_code) -> list]):
                A function that returns a mask array of length len(web_page.nodes)
                indicating whether the node is a valid candidate
            top_k (int): Number of predictions to return
            project (bool): Whether to project the phrase embeddings with a linear layer
        """
        super(EncodingModel, self).__init__()
        self.phrase_embedder = phrase_embedder
        self.node_embedder = node_embedder
        self.node_filter = node_filter

        self.use_neighbors = use_neighbors
        if self.use_neighbors:
            self._max_neighbors = max_neighbors
            self._neighbor_rels = {x: i for (i, x) in enumerate(sorted(set(neighbor_rels)))}
            assert all(x in GraphRels.LOOKUP for x in self._neighbor_rels)

            extra_nodes = len(neighbor_rels) * max_neighbors
        else:
            extra_nodes = 0

        self.encode_dim = node_embedder.embed_dim * (1 + 2 * (1 + extra_nodes))
        self.score = nn.Linear(self.encode_dim, 1)
        self.dropout = nn.Dropout(dropout)
        if project:
            self.proj = nn.Linear(phrase_embedder.embed_dim, node_embedder.embed_dim)
        else:
            assert phrase_embedder.embed_dim == node_embedder.embed_dim
            self.proj = None
        self.loss = nn.CrossEntropyLoss(reduce=False)
        self.top_k = top_k

    def forward(self, web_page, examples, logits_only=False):
        """Compute predictions and loss.

        Args:
            web_page (WebPage): The web page of the examples
            examples (list[PhraseNodeExample]): Must be from the same web page.
        Returns:
            logits (Tensor): num_phrases x num_nodes
                Each entry (i,j) is the logit for p(node_j | phrase_i)
            losses (Tensor): num_phrases
            predictions (Tensor): num_phrases
        """
        # Embed the nodes + normalize
        # num_nodes x dim
        node_embeddings = self.node_embedder(web_page.nodes)
        node_embeddings = node_embeddings / torch.clamp(node_embeddings.norm(p=2, dim=1, keepdim=True), min=1e-8)
        # Embed the phrases + normalize
        phrases = []
        for example in examples:
            phrases.append(word_tokenize(example.phrase.lower()))
        # num_phrases x dim
        phrase_embeddings = self.phrase_embedder(phrases)
        if self.proj is not None:
            phrase_embeddings = F.sigmoid(self.proj(phrase_embeddings))
        else:
            pass
        phrase_embeddings = phrase_embeddings / torch.clamp(phrase_embeddings.norm(p=2, dim=1, keepdim=True), min=1e-8)

        ps = torch.split(phrase_embeddings, 1, dim=0)
        logits = []

        # only loop on phrases
        if not self.use_neighbors:
            for p in ps:
                p = p.expand(node_embeddings.shape[0],-1)
                encoding = torch.cat((p, node_embeddings, p*node_embeddings), dim=1)
                encoding = self.dropout(encoding)
                scores = self.score(encoding)
                logits.append(scores)
        else:
            neighbors, masks = web_page.get_spatial_neighbors()
            neighbors, masks = V(LT(neighbors)), V(FT(masks))
            masks = masks.unsqueeze(dim=2)

            for p in ps:
                batch_size = node_embeddings.shape[0]
                p = p.expand(batch_size,-1)

                neighbor_scores = torch.index_select(node_embeddings, 0,
                        neighbors.view(-1))
                neighbor_scores = neighbor_scores.view(batch_size,
                        neighbors.shape[1], -1)
                neighbor_scores = neighbor_scores * masks

                encoding = [p]
                neighbor_scores = torch.split(neighbor_scores,1,dim=1)
                neighbor_scores = map(lambda x: torch.squeeze(x,dim=1), neighbor_scores)
                for n in [node_embeddings] + neighbor_scores:
                    encoding += [n, p*n]
                encoding = torch.cat(encoding, dim=1)
                encoding = self.dropout(encoding)
                scores = self.score(encoding)
                logits.append(scores)

        logits = torch.cat(logits, dim=1)
        logits = logits.permute(1,0)

        # Filter the candidates
        node_filter_mask = self.node_filter(web_page, examples[0].web_page_code)
        log_node_filter_mask = V(FT([0. if x else -999999. for x in node_filter_mask]))
        logits = logits + log_node_filter_mask
        if logits_only:
            return logits
        # Losses and predictions
        targets = V(LT([web_page.xid_to_ref.get(x.target_xid, 0) for x in examples]))
        mask = V(FT([int(
                x.target_xid in web_page.xid_to_ref
                and node_filter_mask[web_page.xid_to_ref[x.target_xid]]
            ) for x in examples]))
        losses = self.loss(logits, targets) * mask
        #print '=' * 20, examples[0].web_page_code
        #print [node_filter_mask[web_page.xid_to_ref.get(x.target_xid, 0)] for x in examples]
        #print [logits.data[i, web_page.xid_to_ref.get(x.target_xid, 0)] for (i, x) in enumerate(examples)]
        #print logits, targets, mask, losses
        if not np.isfinite(losses.data.sum()):
            #raise ValueError('Losses has NaN')
            logging.warn('Losses has NaN')
            #print losses
        # num_phrases x top_k
        top_k = min(self.top_k, len(web_page.nodes))
        predictions = torch.topk(logits, top_k, dim=1)[1]
        return logits, losses, predictions

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
# Final model

def get_encoding_model(config, node_embedder):
    """Create a new EncodingModel

    Args:
        config (Config): the root config
        node_embedder (NodeEmbedder)
    Returns:
        EncodingModel
    """
    phrase_embedder = node_embedder.utterance_embedder
    node_filter = get_node_filter(config.model.node_filter)
    model = EncodingModel(phrase_embedder, node_embedder, node_filter, config.model.top_k,
            use_neighbors=config.model.use_neighbors,
            dropout=config.model.dropout)
    return model
