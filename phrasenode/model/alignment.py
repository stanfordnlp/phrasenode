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
from phrasenode.utils import word_tokenize, word_tokenize2
from phrasenode.vocab import GloveEmbeddings, RandomEmbeddings, read_frequency_vocab


def embed_tokens(token_embedder, max_words, node_texts):
    """
    Args:
        token_embedder (TokenEmbedder)
        words (TokenEmbedder)
    """
    # Cut to max_words + look up indices
    texts = [text[:max_words-1] + [EOS] for text in node_texts]
    token_indices = SequenceBatch.from_sequences(texts, token_embedder.vocab, min_seq_length=max_words)
    # batch x seq_len x token_embed_dim
    token_embeds = token_embedder.embed_seq_batch(token_indices)
    return token_embeds

    # words = words[:max_words] + [EOS]
    # # seq_len x token_embed_dim
    # return token_embedder.embed_tokens(words)


def semantic_attrs(attrs):
    whitelist = ['aria','tooltip','placeholder','label','title','name','alt']
    attrs = [value for key, value in attrs.items() if any(k in key.lower() for k in whitelist)]
    return ' '.join(attrs)


class AlignmentModel(nn.Module):

    def __init__(self, phrase_embedder, token_embedder,
            max_words, node_filter, top_k=5, dropout=0.3,
            ablate_text=False, ablate_attrs=False, use_neighbors=False, use_tags=False,
            neighbor_rels=['above','left'], max_neighbors=1):
            # neighbor_rels=['above','below','left','right'], max_neighbors=1):
        """
        Args:
            node_filter (callable[(WebPage, web_page_code) -> list]):
                A function that returns a mask array of length len(web_page.nodes)
                indicating whether the node is a valid candidate
            top_k (int): Number of predictions to return
        """
        super(AlignmentModel, self).__init__()

        self.phrase_embedder = phrase_embedder

        self.ablate_text = ablate_text
        self.ablate_attrs = ablate_attrs
        self.use_neighbors = use_neighbors

        conv_dim = 3
        dilation = 2
        pool_dim = 2
        # doesn't change the dimension
        self.conv2d = nn.Conv2d(1, 1, conv_dim, padding=conv_dim-1)
        self.conv2d_dilated = nn.Conv2d(1, 1, conv_dim, padding=conv_dim-1, dilation=dilation)
        self.pooler = nn.MaxPool2d(pool_dim)
        self.score_dim = int(math.pow(math.ceil((max_words+1) / float(pool_dim)), 2))
        self.scorer = nn.Linear(self.score_dim, 1)

        # idea: compute a bunch of latent score vectors before computing
        # logits, take a linear layer down to 1 score
        # purpose: if you want to compute scores with neighbors, you can now
        # average neighbor score vectors and Linear down to 1 score
        neighbor_score_dim = 10

        if self.use_neighbors:
            self._max_neighbors = max_neighbors
            self._neighbor_rels = {x: i for (i, x) in enumerate(sorted(set(neighbor_rels)))}
            self.num_rels = len(neighbor_rels)
            assert all(x in GraphRels.LOOKUP for x in self._neighbor_rels)

            # score_embed_dim = int(math.ceil((self.score_dim) / float(pool_dim)))
            score_dim = self.score_dim * (self.num_rels*max_neighbors + 1)
            # self.pool_neighbors = nn.MaxPool1d(pool_dim)
            self._final_neighbor_linear = nn.Linear(score_dim, 1)
            extra_nodes = self.num_rels * max_neighbors
        else:
            extra_nodes = 0

        self.dropout = nn.Dropout(dropout)

        self.token_embedder = token_embedder
        self.max_words = max_words
        self.node_filter = node_filter
        self.loss = nn.CrossEntropyLoss(reduce=False)
        self.top_k = top_k

        self.use_tags = use_tags
        if self.use_tags:
            tags = [UNK, EOS] + TAGS
            tag_dim = 10
            self._tag_embedder = TokenEmbedder(RandomEmbeddings(tags, tag_dim))
            self.project_tag = nn.Linear(tag_dim + self.score_dim, self.score_dim)

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
        def max_scorer(pairwise_scores):
            """
            Args:
                pairwise_scores: num_nodes x phrase_len x max_text_len
            """
            scores = torch.max(pairwise_scores, dim=1)[0]
            return torch.max(scores, dim=1)[0]
        def cnn_scorer(pairwise_scores):
            """
            Args:
                pairwise_scores: num_nodes x phrase_len x max_text_len
            """
            scores = torch.unsqueeze(pairwise_scores, dim=1)
            scores = self.conv2d(scores)
            scores = self.conv2d_dilated(scores)
            scores = self.pooler(scores)
            scores = torch.squeeze(scores, dim=1)
            # dim = scores.shape[1]*scores.shape[2]
            scores = scores.view(-1,self.score_dim)
            if self.use_tags:
                tags = [node.tag for node in web_page.nodes]
                tag_embeddings = self._tag_embedder.embed_tokens(tags)
                scores = torch.cat((scores,tag_embeddings), dim=1)
                scores = self.project_tag(scores)
            scores = self.scorer(scores)
            scores = torch.squeeze(scores, dim=1)
            return scores
        def neighbor_cnn_scorer(pairwise_scores):
            """
            Args:
                pairwise_scores: num_nodes x phrase_len x max_text_len
            """
            scores = torch.unsqueeze(pairwise_scores, dim=1)
            scores = self.conv2d(scores)
            scores = self.conv2d_dilated(scores)
            scores = self.pooler(scores)
            scores = torch.squeeze(scores, dim=1)
            # dim = scores.shape[1]*scores.shape[2]
            scores = scores.view(-1,self.score_dim)
            if self.use_tags:
                tags = [node.tag for node in web_page.nodes]
                tag_embeddings = self._tag_embedder.embed_tokens(tags)
                scores = torch.cat((scores,tag_embeddings), dim=1)
                scores = self.project_tag(scores)
            return scores

        # Tokenize the nodes
        # num_nodes x text_length x embed_dim
        texts = []
        for node in web_page.nodes:
            text = ' '.join(node.all_texts(max_words=self.max_words))
            output = []
            if not self.ablate_text:
                output += word_tokenize2(text)
            if not self.ablate_attrs:
                # TODO better way to include attributes?
                output += word_tokenize2(semantic_attrs(node.attributes))
            texts.append(output)

        embedded_texts = embed_tokens(self.token_embedder, self.max_words, texts)
        embedded_texts_values = self.dropout(embedded_texts.values)

        embedded_texts = embedded_texts_values * embedded_texts.mask.unsqueeze(2)

        # Tokenize the phrases
        # num_phrases x phrase_length x embed_dim
        logits = []

        if not self.use_neighbors:
            for example in examples:
                phrase = [word_tokenize2(example.phrase)]
                embedded_phrase = embed_tokens(self.token_embedder, self.max_words, phrase)

                embedded_phrase_values = self.dropout(embedded_phrase.values)

                # expand: num_nodes x phrase_len x embed_dim
                batch_phrase = embedded_phrase_values.expand(len(texts),-1,-1)
                # permute embedded_texts: num_nodes x embed_dim x max_text_len
                pairwise_scores = torch.bmm(batch_phrase, embedded_texts.permute(0,2,1))

                # compute scores
                scores = cnn_scorer(pairwise_scores)
                logits.append(torch.unsqueeze(scores, dim=0))
        else:
            intermediate_scores = []
            for example in examples:
                phrase = [word_tokenize2(example.phrase)]
                embedded_phrase = embed_tokens(self.token_embedder, self.max_words, phrase)

                embedded_phrase_values = self.dropout(embedded_phrase.values)

                # expand: num_nodes x phrase_len x embed_dim
                batch_phrase = embedded_phrase_values.expand(len(texts),-1,-1)
                # permuted embedded_texts: num_nodes x embed_dim x max_text_len
                pairwise_scores = torch.bmm(batch_phrase, embedded_texts.permute(0,2,1))
                node_score = neighbor_cnn_scorer(pairwise_scores)
                intermediate_scores.append(node_score)

            neighbors, masks = web_page.get_spatial_neighbors()
            neighbors, masks = V(LT(neighbors)), V(FT(masks))
            masks = masks.unsqueeze(dim=2)

            # each node_score tensor is parameterized by phrase
            for node_score in intermediate_scores:
                # get pairwise_scores for all neighbors...
                # neighbors, rels = self._get_neighbors(web_page)
                batch_size = len(node_score)
                neighbor_scores = torch.index_select(node_score, 0,
                        neighbors.view(-1))
                neighbor_scores = neighbor_scores.view(batch_size,
                        neighbors.shape[1], -1)
                neighbor_scores = neighbor_scores * masks

                if neighbor_scores.shape[1] < self.num_rels:
                    more = self.num_rels - neighbor_scores.shape[1]
                    num_nodes, _, embed_dim = neighbor_scores.shape
                    padding = V(torch.zeros(num_nodes, more, embed_dim))
                    neighbor_scores = torch.cat((neighbor_scores,padding), dim=1)
                # num_nodes x num_neighbors x intermediate_score_dim

                node_score = torch.unsqueeze(node_score, dim=1)
                scores = torch.cat((node_score,neighbor_scores), dim=1)

                scores = scores.view(node_score.shape[0],-1)
                scores = self._final_neighbor_linear(scores)
                scores = torch.squeeze(scores, dim=1)

                logits.append(torch.unsqueeze(scores, dim=0))

        logits = torch.cat(logits, dim=0)

        # Filter the candidates
        node_filter_mask = self.node_filter(web_page, examples[0].web_page_code) # what does this do?
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

def get_alignment_model(config, node_embedder):
    """Create a new AlignmentModel

    Args:
        config (Config): the root config
        node_embedder (NodeEmbedder)
    Returns:
        AlignmentModel
    """
    cm = config.model
    cmu = cm.utterance_embedder

    #glove_embeddings = GloveEmbeddings(cmu.vocab_size, cmu.glove_dim)
    #token_embedder = TokenEmbedder(glove_embeddings, trainable=cmu.trainable)

    phrase_embedder = node_embedder.utterance_embedder
    token_embedder = phrase_embedder.token_embedder

    node_filter = get_node_filter(cm.node_filter)
    model = AlignmentModel(phrase_embedder, token_embedder,
            cmu.max_words, node_filter, cm.top_k,
            dropout=cm.dropout,
            ablate_text=cm.ablate_text,
            ablate_attrs=cm.ablate_attrs,
            use_neighbors=cm.use_neighbors,
            use_tags=cm.use_tags)
    return model
