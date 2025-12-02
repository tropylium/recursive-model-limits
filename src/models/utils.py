from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets)


class RecursiveModel(FeedForwardModel):
    """Base class for recursive models that maintain state across iterations."""

    @abstractmethod
    def init_state(self, batch_size: int) -> Any:
        """Initialize the state for a batch of images."""
        pass

    @abstractmethod
    def forward(self, img: torch.Tensor, prev_state: Any) -> tuple[torch.Tensor, Any]:
        """
        Forward pass with state.

        Args:
            img: Input images [B, C, H, W]
            prev_state: Previous state from last iteration

        Returns:
            logits: Classification logits [B, num_classes]
            new_state: Updated state for next iteration
        """
        pass
