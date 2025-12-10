from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.utils import RecursiveModel
from src.utils.checkpoint import save_checkpoint
from src.utils.distributed import (get_rank, get_world_size, is_main_process,
                                   reduce_tensor)
from src.utils.logging import log_metrics, print_rank_0


class Trainer:
    """
    Trainer class for distributed training with AMP and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        checkpoint_dir: Path,
        use_amp: bool = True,
        checkpoint_every: int = 10,
        rank: int = 0,
        world_size: int = 1,
        best_val_acc: float = 0.0,
    ):
        """
        Initialize the Trainer.

        Args:
            model: Model to train (will be wrapped in DDP if world_size > 1)
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            use_amp: Whether to use automatic mixed precision
            checkpoint_every: Save checkpoint every N epochs
            rank: Process rank
            world_size: Total number of processes
            best_val_acc: Best validation accuracy so far (for resuming)
        """
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every

        # Wrap model in DDP if distributed
        if world_size > 1:
            self.model = DDP(
                model,
                device_ids=[rank],
                find_unused_parameters=True,  # Required for models with conditional paths
            )
        else:
            self.model = model

        # Detect if model supports stateful/recursive training
        # Check the underlying model (unwrap DDP if needed)
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        self.is_recursive_model = isinstance(base_model, RecursiveModel)

        self.optimizer = optimizer
        self.scheduler = scheduler

        # AMP setup
        self.use_amp = use_amp
        self.scaler = GradScaler("cuda") if use_amp else None

        # Track best validation metric
        self.best_val_acc = best_val_acc

    @classmethod
    def from_train_state(cls, state: "TrainState") -> "Trainer":
        """
        Create a Trainer from a TrainState object.

        Args:
            state: TrainState containing all training components

        Returns:
            Initialized Trainer
        """
        return cls(
            model=state.model,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            device=state.device,
            checkpoint_dir=state.checkpoint_dir,
            use_amp=state.config.training.use_amp,
            checkpoint_every=state.config.training.checkpoint_every,
            rank=state.rank,
            world_size=state.world_size,
            best_val_acc=state.best_val_acc,
        )

    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Helper to compute loss, handling DDP wrapping.

        Args:
            output: Model output (logits)
            target: Target labels

        Returns:
            Computed loss
        """
        if hasattr(self.model, "module"):
            return self.model.module.compute_loss(output, target)
        else:
            return self.model.compute_loss(output, target)

    def _forward_standard(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Standard forward pass for feedforward models.

        Args:
            data: Input data
            target: Target labels

        Returns:
            loss: Computed loss
            output: Model output (logits)
        """
        if self.use_amp:
            with autocast("cuda"):
                output = self.model(data)
                loss = self._compute_loss(output, target)
        else:
            output = self.model(data)
            loss = self._compute_loss(output, target)

        return loss, output

    def _train_recursive(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> tuple[float, torch.Tensor]:
        """
        Training pass for recursive models with per-step backpropagation.
        Each recursion step computes loss and backprops independently.

        Args:
            data: Input data
            target: Target labels

        Returns:
            avg_loss: Average loss across all recursion steps (as float, for logging)
            logits: Final output logits (for metrics)
        """
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        state = base_model.init_state(data.size(0))

        total_loss = 0.0

        for step in range(base_model.config.max_recursion_steps):
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast("cuda"):
                    logits, state = self.model(data, state)
                    step_loss = self._compute_loss(logits, target)
                self.scaler.scale(step_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, state = self.model(data, state)
                step_loss = self._compute_loss(logits, target)
                step_loss.backward()
                self.optimizer.step()

            total_loss += step_loss.item()

        avg_loss = total_loss / base_model.config.max_recursion_steps
        return avg_loss, logits

    @torch.no_grad()
    def _forward_recursive_eval(
        self, data: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluation forward pass for recursive models.
        Runs through all recursion steps and returns final prediction.

        Args:
            data: Input data
            target: Target labels

        Returns:
            avg_loss: Average loss across all recursion steps
            logits: Final output logits (for metrics)
        """
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        state = base_model.init_state(data.size(0))

        total_loss = 0.0

        for step in range(base_model.config.max_recursion_steps):
            if self.use_amp:
                with autocast("cuda"):
                    logits, state = self.model(data, state)
                    step_loss = self._compute_loss(logits, target)
            else:
                logits, state = self.model(data, state)
                step_loss = self._compute_loss(logits, target)

            total_loss = total_loss + step_loss

        avg_loss = total_loss / base_model.config.max_recursion_steps
        return avg_loss, logits

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        # Only show progress bar on rank 0
        if is_main_process(self.rank):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        else:
            pbar = train_loader

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Route to appropriate training method based on model type
            if self.is_recursive_model:
                # Recursive: per-step backprop handled inside _train_recursive
                loss_value, output = self._train_recursive(data, target)
            else:
                # Standard: forward then backward
                self.optimizer.zero_grad()
                loss, output = self._forward_standard(data, target)

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                loss_value = loss.item()

            total_loss += loss_value
            num_batches += 1

            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

            # Update progress bar
            if is_main_process(self.rank) and isinstance(pbar, tqdm):
                current_acc = 100.0 * correct / total if total > 0 else 0.0
                pbar.set_postfix({"loss": loss_value, "acc": f"{current_acc:.2f}%"})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        # Aggregate metrics across all processes if distributed
        if self.world_size > 1:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            correct_tensor = torch.tensor([correct], device=self.device)
            total_tensor = torch.tensor([total], device=self.device)

            avg_loss = reduce_tensor(loss_tensor, average=True).item()
            correct = reduce_tensor(correct_tensor, average=False).item()
            total = reduce_tensor(total_tensor, average=False).item()
            accuracy = 100.0 * correct / total if total > 0 else 0.0

        return {'train_loss': avg_loss, 'train_accuracy': accuracy}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        # Only show progress bar on rank 0
        if is_main_process(self.rank):
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        else:
            pbar = val_loader

        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)

            # Route to appropriate forward method based on model type
            if self.is_recursive_model:
                loss, output = self._forward_recursive_eval(data, target)
            else:
                loss, output = self._forward_standard(data, target)

            total_loss += loss.item()

            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        # Aggregate metrics across all processes
        if self.world_size > 1:
            loss_tensor = torch.tensor([avg_loss], device=self.device)
            acc_tensor = torch.tensor([accuracy], device=self.device)
            correct_tensor = torch.tensor([correct], device=self.device)
            total_tensor = torch.tensor([total], device=self.device)

            avg_loss = reduce_tensor(loss_tensor, average=True).item()
            correct = reduce_tensor(correct_tensor, average=False).item()
            total = reduce_tensor(total_tensor, average=False).item()
            accuracy = 100.0 * correct / total if total > 0 else 0.0

        return {'val_loss': avg_loss, 'val_accuracy': accuracy}

    def save_checkpoint_if_needed(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Save checkpoint if needed (periodic or best model).
        Only saves on rank 0.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary with current metrics
        """
        if not is_main_process(self.rank):
            return

        val_acc = metrics.get('val_accuracy', 0.0)

        # Save periodic checkpoint
        if (epoch + 1) % self.checkpoint_every == 0:
            filename = f"checkpoint_epoch_{epoch + 1}.pt"
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch + 1,
                checkpoint_dir=self.checkpoint_dir,
                filename=filename,
                best_metric=self.best_val_acc,
            )
            print_rank_0(f"Saved checkpoint: {filename}", self.rank)

        # Save best model
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            filename = "best_model.pt"
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch + 1,
                checkpoint_dir=self.checkpoint_dir,
                filename=filename,
                best_metric=self.best_val_acc,
            )
            print_rank_0(f"New best model! Val Acc: {val_acc:.2f}%", self.rank)

    def log_metrics_to_wandb(self, metrics: Dict[str, float], epoch: int) -> None:
        """
        Log metrics to wandb (only on rank 0).
        
        Args:
            metrics: Dictionary with metrics to log
            epoch: Current epoch number
        """
        if not is_main_process(self.rank):
            return

        # Add learning rate to metrics
        if self.scheduler is not None:
            metrics['learning_rate'] = self.scheduler.get_last_lr()[0]
        else:
            metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

        log_metrics(metrics, step=epoch, rank=self.rank)

    def step_scheduler(self) -> None:
        """Step the learning rate scheduler."""
        if self.scheduler is not None:
            self.scheduler.step()
