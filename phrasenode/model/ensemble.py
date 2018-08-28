"""An ensemble between encoding and alignment models."""

import numpy as np

import torch
import torch.optim as optim
from torch import LongTensor as LT, FloatTensor as FT
import torch.nn as nn
import torch.nn.functional as F

from gtd.ml.torch.utils import GPUVariable as V


class EnsembleModel(nn.Module):

    def __init__(self, encoding_model, alignment_model, node_filter, top_k=5):
        super(EnsembleModel, self).__init__()
        self._encoding_model = encoding_model
        self._alignment_model = alignment_model
        self._weight = V(FT([1.0, 1.0]))

        self.node_filter = node_filter
        self.loss = nn.CrossEntropyLoss(reduce=False)
        self.top_k = top_k

    def forward(self, web_page, examples):
        e_logits = self._encoding_model(web_page, examples, logits_only=True)
        a_logits = self._alignment_model(web_page, examples, logits_only=True)

        # Normalize
        e_logprobs = F.log_softmax(e_logits, dim=1)
        a_logprobs = F.log_softmax(a_logits, dim=1)

        logits = e_logprobs * self._weight[0] + a_logprobs * self._weight[1]

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

def get_ensemble_model(config, node_embedder):
    """Create a new EnsembleModel.

    Args:
        config (Config): the root config
        node_embedder (NodeEmbedder)
    Returns:
        AlignmentModel
    """
    from phrasenode.model.encoding import get_encoding_model
    from phrasenode.model.alignment import get_alignment_model

    encoding_model = get_encoding_model(config, node_embedder)
    alignment_model = get_alignment_model(config, node_embedder)

    model = EnsembleModel(encoding_model, alignment_model,
            encoding_model.node_filter, config.model.top_k,
            )
    return model
