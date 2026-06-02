"""
Monte Carlo Dropout for epistemic uncertainty estimation.
"""
import torch
import torch.nn as nn
import numpy as np
from .replay_buffer import ReplayBuffer


def add_dropout_to_backbone(backbone: nn.Module, p: float = 0.1):
    """Add dropout layers to each TransformerBlock FFN."""
    for block in backbone.blocks:
        if not any(isinstance(m, nn.Dropout) for m in block.ffn.children()):
            block.ffn.add_module('extra_dropout', nn.Dropout(p))


def mc_uncertainty(model: nn.Module, x: torch.Tensor,
                   n_passes: int = 20) -> torch.Tensor:
    """
    Estimate epistemic uncertainty via Monte Carlo Dropout.
    Returns entropy of the mean predictive distribution.
    """
    model.train()
    probs_list = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(x)
            probs_list.append(torch.softmax(logits, dim=-1))
    model.eval()
    mean_probs = torch.stack(probs_list).mean(dim=0)
    entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
    return entropy


class MCDropoutReplayBuffer:
    """Wrapper around ReplayBuffer that uses MC Dropout for priority scoring."""

    def __init__(self, buffer: ReplayBuffer, n_mc_passes: int = 20,
                 epsilon: float = 0.1):
        self.buffer = buffer
        self.n_mc_passes = n_mc_passes
        self.epsilon = epsilon

    def sample_uncertain(self, model: nn.Module, batch_size: int,
                         device: torch.device):
        """Sample examples proportional to their uncertainty scores."""
        if len(self.buffer) < batch_size:
            return self.buffer.sample(batch_size, device)

        X, y, scores = self.buffer.buffer_X, self.buffer.buffer_y, self.buffer.priority_scores
        size = self.buffer.size
        probs = (scores[:size] + self.epsilon)
        probs = probs / probs.sum()
        indices = np.random.choice(size, size=min(batch_size, size),
                                   replace=False, p=probs)
        Xb = torch.from_numpy(X[indices]).to(device)
        yb = torch.from_numpy(y[indices]).to(device)
        return Xb, yb
