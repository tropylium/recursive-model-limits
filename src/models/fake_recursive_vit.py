import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.recursive_vit import RecursiveViTConfig, RecursiveViTInner


class FakeRecursiveViT(nn.Module):
    def __init__(self, config: RecursiveViTConfig):
        super().__init__()
        self.config = config

        self.inners = nn.ModuleList([RecursiveViTInner(config) for _ in range(self.config.max_recursion_steps)])

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        state = self.inners[0].init_state(img.size(0))
        for inner in self.inners:
            logits, state = inner(img, state)
        return logits

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets)