from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.trainer import Trainer
from src.utils.logging import print_rank_0


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    config: DictConfig,
    device: torch.device,
    checkpoint_dir: Path,
    rank: int = 0,
    world_size: int = 1,
) -> None:
    """
    Main training loop.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config: Hydra config object
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        rank: Process rank
        world_size: Total number of processes
    """
    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        use_amp=config.training.use_amp,
        checkpoint_every=config.training.checkpoint_every,
        rank=rank,
        world_size=world_size,
    )
    
    print_rank_0("=" * 80, rank)
    print_rank_0(f"Starting training for {config.training.epochs} epochs", rank)
    print_rank_0(f"Batch size per GPU: {config.training.batch_size}", rank)
    print_rank_0(f"Total batch size: {config.training.batch_size * world_size}", rank)
    print_rank_0(f"Using AMP: {config.training.use_amp}", rank)
    print_rank_0(f"Validation every {config.training.checkpoint_every} epochs", rank)
    print_rank_0("=" * 80, rank)
    
    # Training loop
    for epoch in range(config.training.epochs):
        # Train for one epoch
        train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Check if this is a checkpoint epoch or the last epoch
        is_checkpoint_epoch = (epoch + 1) % config.training.checkpoint_every == 0
        is_last_epoch = epoch == config.training.epochs - 1
        should_validate = is_checkpoint_epoch or is_last_epoch
        
        # Validate only on checkpoint epochs
        if should_validate:
            val_metrics = trainer.validate(val_loader, epoch)
            metrics = {**train_metrics, **val_metrics}
            
            # Print metrics with validation
            print_rank_0(
                f"Epoch {epoch + 1}/{config.training.epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.2f}%",
                rank
            )
        else:
            metrics = train_metrics
            
            # Print metrics without validation
            print_rank_0(
                f"Epoch {epoch + 1}/{config.training.epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_accuracy']:.2f}%",
                rank
            )
        
        # Log to wandb
        trainer.log_metrics_to_wandb(metrics, epoch)
        
        # Step scheduler
        trainer.step_scheduler()
        
        # Save checkpoint if needed (only on checkpoint epochs)
        if should_validate:
            trainer.save_checkpoint_if_needed(epoch, metrics)
    
    print_rank_0("=" * 80, rank)
    print_rank_0(f"Training complete! Best Val Acc: {trainer.best_val_acc:.2f}%", rank)
    print_rank_0("=" * 80, rank)

