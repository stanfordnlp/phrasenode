"""p(node|text) prediction model based on dot product of embeddings."""
import logging

import numpy as np

import torch
import torch.optim as optim
from torch import LongTensor as LT, FloatTensor as FT
import torch.nn as nn
import torch.nn.functional as F

from gtd.ml.torch.token_embedder import TokenEmbedder
from gtd.ml.torch.utils import GPUVariable as V
from gtd.ml.torch.utils import try_gpu

from phrasenode.constants import UNK, EOS, HIDDEN, TAGS, GraphRels
from phrasenode.node_filter import get_node_filter
from phrasenode.utterance_embedder import AverageUtteranceEmbedder, LSTMUtteranceEmbedder
from phrasenode.utils import word_tokenize
from phrasenode.vocab import GloveEmbeddings, RandomEmbeddings, read_frequency_vocab


class EmbeddingModel(nn.Module):

    def __init__(self, phrase_embedder, node_embedder, node_filter, top_k=5, project=False):
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
        super(EmbeddingModel, self).__init__()
        self.phrase_embedder = phrase_embedder
        self.node_embedder = node_embedder
        self.node_filter = node_filter
        if project:
            self.proj = nn.Linear(phrase_embedder.embed_dim, node_embedder.embed_dim)
        else:
            assert phrase_embedder.embed_dim == node_embedder.embed_dim
            self.proj = None
        self.loss = nn.CrossEntropyLoss(reduce=False)
        self.top_k = top_k

    def forward(self, web_page, examples):
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
        # Dot product
        # num_phrases x num_nodes
        logits = torch.mm(phrase_embeddings, node_embeddings.t())
        # Filter the candidates
        node_filter_mask = self.node_filter(web_page, examples[0].web_page_code)
        log_node_filter_mask = V(FT([0. if x else -999999. for x in node_filter_mask]))
        logits = logits + log_node_filter_mask
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


################################################
# Final model

def get_embedding_model(config, node_embedder):
    """Create a new EmbeddingModel

    Args:
        config (Config): the root config
        node_embedder (NodeEmbedder)
    Returns:
        EmbeddingModel
    """
    phrase_embedder = node_embedder.utterance_embedder
    node_filter = get_node_filter(config.model.node_filter)
    model = EmbeddingModel(phrase_embedder, node_embedder, node_filter, config.model.top_k)
    return model
