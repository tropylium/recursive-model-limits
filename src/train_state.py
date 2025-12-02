from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.utils.checkpoint import load_checkpoint


@dataclass
class TrainState:
    """
    Encapsulates all training state and configuration.
    
    This dataclass contains all components needed for training:
    - Model, optimizer, scheduler
    - Data loaders
    - Configuration
    - Training state (epoch, best metrics, etc.)
    """
    
    # Model & Training Components
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    
    # Data Components
    train_loader: DataLoader
    val_loader: DataLoader
    
    # Configuration
    config: DictConfig
    
    # Training State
    device: torch.device
    checkpoint_dir: Path
    rank: int
    world_size: int
    start_epoch: int = 0
    best_val_acc: float = 0.0
    
    # Trainer (set after initialization)
    trainer: Optional[object] = None
    
    def load_from_checkpoint(self, checkpoint_path: Path) -> "TrainState":
        """
        Load model, optimizer, and scheduler state from a checkpoint.
        Returns a new TrainState with updated state.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Updated TrainState with loaded checkpoint
        """
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint not found at {checkpoint_path}, starting from scratch")
            return self
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint_data = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=str(self.device),
        )
        
        # Update training state
        self.start_epoch = checkpoint_data["epoch"]
        self.best_val_acc = checkpoint_data.get("best_metric", 0.0) or 0.0
        
        print(f"Resumed from epoch {self.start_epoch}, best val acc: {self.best_val_acc:.2f}%")
        
        return self
    
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0
    
    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.world_size > 1

