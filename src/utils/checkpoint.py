import os
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    checkpoint_dir: Path,
    filename: str,
    best_metric: Optional[float] = None,
    **extra_state
) -> None:
    """
    Save a training checkpoint.
    
    Args:
        model: Model to save (will extract state_dict, handling DDP)
        optimizer: Optimizer to save
        scheduler: Learning rate scheduler to save (optional)
        epoch: Current epoch number
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
        best_metric: Best validation metric so far (optional)
        **extra_state: Additional state to save in checkpoint
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract model state_dict (handle DDP wrapper)
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        **extra_state
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into (will handle DDP)
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint to
        
    Returns:
        Dictionary with checkpoint metadata (epoch, best_metric, etc.)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state (handle DDP wrapper)
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'best_metric': checkpoint.get('best_metric', None),
    }


def get_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory to search
        
    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        return None
    
    # Sort by epoch number
    checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
    return checkpoints[-1]

