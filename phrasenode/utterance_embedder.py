"""Utterance embedder"""
import torch
from torch import LongTensor as LT, FloatTensor as FT
import torch.nn as nn
import torch.nn.functional as F

from gtd.ml.torch.attention import Attention
from gtd.ml.torch.seq_batch import SequenceBatch
from gtd.ml.torch.source_encoder import BidirectionalSourceEncoder
from gtd.ml.torch.utils import GPUVariable as V

from phrasenode.constants import EOS


################################################
# Utterance Embedder

class AverageUtteranceEmbedder(nn.Module):
    """Takes a string, embeds the tokens using the token_embedder,
    and return the average of the results.
    """

    def __init__(self, token_embedder, max_words):
        """Initialize

        Args:
            token_embedder (TokenEmbedder): used to embed each token
            max_words (int): maximum number of words to embed
        """
        super(AverageUtteranceEmbedder, self).__init__()
        self._token_embedder = token_embedder
        self._embed_dim = token_embedder.embed_dim
        self._max_words = max_words

    def forward(self, utterances):
        """Embeds an utterances.

        Args:
            utterances (list[list[unicode]]): list[unicode] is a list of tokens
            forming a sentence. list[list[unicode]] is batch of sentences.

        Returns:
            Variable[FloatTensor]: batch x glove_dim
                (average of glove vectors)
        """
        # Cut to max_words + look up indices
        utterances = [utterance[:self._max_words] + [EOS] for utterance in utterances]
        token_indices = SequenceBatch.from_sequences(
                utterances, self._token_embedder.vocab)
        # batch x seq_len x token_embed_dim
        token_embeds = self._token_embedder.embed_seq_batch(token_indices)
        # batch x token_embed_dim
        averaged = SequenceBatch.reduce_mean(token_embeds)
        return averaged

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def max_words(self):
        return self._max_words

    @property
    def token_embedder(self):
        return self._token_embedder


class LSTMUtteranceEmbedder(nn.Module):
    """Takes a string, embeds the tokens using the token_embedder, and passes
    the embeddings through a biLSTM padded / masked up to sequence_length.
    Returns the concatenation of the two front and end hidden states.
    """

    def __init__(self, token_embedder, lstm_dim, max_words):
        """Initialize

        Args:
            token_embedder (TokenEmbedder): used to embed each token
            lstm_dim (int): output dim of the lstm
            max_words (int): maximum number of words to embed
        """
        super(LSTMUtteranceEmbedder, self).__init__()
        self._token_embedder = token_embedder
        self._bilstm = BidirectionalSourceEncoder(
               token_embedder.embed_dim, lstm_dim, nn.LSTMCell)
        self._embed_dim = lstm_dim
        self._max_words = max_words

    def forward(self, utterances):
        """Embeds a batch of utterances.

        Args:
            utterances (list[list[unicode]]): list[unicode] is a list of tokens
            forming a sentence. list[list[unicode]] is batch of sentences.

        Returns:
            Variable[FloatTensor]: batch x lstm_dim
                (concatenated first and last hidden states)
        """
        # Cut to max_words + look up indices
        utterances = [utterance[:self._max_words] + [EOS] for utterance in utterances]
        token_indices = SequenceBatch.from_sequences(
                utterances, self._token_embedder.vocab)
        # batch x seq_len x token_embed_dim
        token_embeds = self._token_embedder.embed_seq_batch(token_indices)
        bi_hidden_states = self._bilstm(token_embeds.split())
        final_states = torch.cat(bi_hidden_states.final_states, 1)
        return torch.stack(final_states, 0)

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def max_words(self):
        return self._max_words

    @property
    def token_embedder(self):
        return self._token_embedder



class AttentionUtteranceEmbedder(nn.Module):
    """Takes a string, embeds the tokens using the token_embedder, and passes
    the embeddings through a biLSTM padded / masked up to sequence_length.
    Returns the concatenation of the two front and end hidden states.
    """

    def __init__(self, token_embedder, lstm_dim, max_words):
        """Initialize

        Args:
            token_embedder (TokenEmbedder): used to embed each token
            lstm_dim (int): output dim of the lstm
            max_words (int): maximum number of words to embed
        """
        super(AttentionUtteranceEmbedder, self).__init__()
        self._token_embedder = token_embedder
        self._bilstm = BidirectionalSourceEncoder(
               token_embedder.embed_dim, lstm_dim, nn.LSTMCell)
        self._embed_dim = lstm_dim
        self._max_words = max_words

        self._attention = Attention(token_embedder.embed_dim, lstm_dim, lstm_dim)

    def forward(self, utterances):
        """Embeds a batch of utterances.

        Args:
            utterances (list[list[unicode]]): list[unicode] is a list of tokens
            forming a sentence. list[list[unicode]] is batch of sentences.

        Returns:
            Variable[FloatTensor]: batch x lstm_dim
                (concatenated first and last hidden states)
        """
        # Cut to max_words + look up indices
        utterances = [utterance[:self._max_words] + [EOS] for utterance in utterances]
        token_indices = SequenceBatch.from_sequences(
                utterances, self._token_embedder.vocab)
        # batch x seq_len x token_embed_dim
        token_embeds = self._token_embedder.embed_seq_batch(token_indices)
        # print('token_embeds', token_embeds)
        bi_hidden_states = self._bilstm(token_embeds.split())
        final_states = torch.cat(bi_hidden_states.final_states, 1)

        hidden_states = SequenceBatch.cat(bi_hidden_states.combined_states)
        return self._attention(hidden_states, final_states).context

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def max_words(self):
        return self._max_words

    @property
    def token_embedder(self):
        return self._token_embedder

