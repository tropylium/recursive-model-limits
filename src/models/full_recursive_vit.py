import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.recursive_vit import (RecursiveViTConfig, RecursiveViTInner,
                                      RecursiveViTState)
from src.models.utils import RecursiveModel


class FullRecursiveViT(RecursiveModel):
    """
    Recursive ViT that exposes state for external management.
    Each forward call performs ONE step through the transformer.
    The training loop handles iterating through multiple steps.
    """
    def __init__(self, config: RecursiveViTConfig):
        super().__init__()
        self.config = config
        self.inner = RecursiveViTInner(config)

    def init_state(self, batch_size: int) -> RecursiveViTState:
        """Initialize state for a batch of images."""
        return self.inner.init_state(batch_size)

    def forward(self, img: torch.Tensor, prev_state: RecursiveViTState) -> tuple[torch.Tensor, RecursiveViTState]:
        """
        Single recursive step through the transformer.
        
        Args:
            img: Input images [B, C, H, W]
            prev_state: Previous state from last iteration
            
        Returns:
            logits: Classification logits [B, num_classes]
            new_state: Updated state for next iteration
        """
        # Perform one step through the inner transformer
        logits, new_state = self.inner(img, prev_state)
        return logits, RecursiveViTState(step=new_state.step, state=new_state.state.detach())