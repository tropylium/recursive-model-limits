from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.datasets import dataset_factory
from src.models.loader import load_model_class
from src.utils.checkpoint import load_checkpoint
from src.utils.logging import print_rank_0


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
    train_dataset: object  # Keep reference to dataset for class names, etc.
    val_dataset: object

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

    @classmethod
    def from_config(
        cls,
        config: DictConfig,
        rank: int,
        world_size: int,
        local_rank: int,
        checkpoint_dir: Optional[Path] = None,
    ) -> "TrainState":
        """
        Create a TrainState from configuration for a new training run.

        Args:
            config: Hydra configuration object
            rank: Process rank
            world_size: Total number of processes
            local_rank: Local rank (GPU index)
            checkpoint_dir: Optional checkpoint directory path. If None, will try to
                          get from Hydra or use current directory.

        Returns:
            Initialized TrainState ready for training
        """
        # Set device
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )

        # Load dataset
        print_rank_0(f"Loading dataset: {config.dataset.name}", rank)
        dataset_fn = dataset_factory(config.dataset.name)
        train_dataset = dataset_fn(train=True)
        val_dataset = dataset_fn(train=False)

        # Create data loaders
        if world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
            train_shuffle = False
        else:
            train_sampler = None
            train_shuffle = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=config.training.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            pin_memory=True,
        )

        print_rank_0(f"Train dataset size: {len(train_dataset)}", rank)
        print_rank_0(f"Val dataset size: {len(val_dataset)}", rank)

        # Load model
        print_rank_0(f"Loading model: {config.arch.model}", rank)
        ModelClass = load_model_class(config.arch.model)
        model = ModelClass(config.arch.config).to(device)

        # Print model info
        if rank == 0:
            num_params = sum(p.numel() for p in model.parameters())
            num_trainable = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            print(f"Model parameters: {num_params:,} (trainable: {num_trainable:,})")

        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

        # Create scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
            eta_min=config.scheduler.eta_min,
        )

        # Setup checkpoint directory
        if checkpoint_dir is None:
            # Try to get from Hydra if running in Hydra context
            try:
                hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
                checkpoint_dir = hydra_output_dir / "checkpoints"
            except ValueError:
                # Not in Hydra context (e.g., running from notebook)
                # Use a temporary directory or current directory
                checkpoint_dir = Path.cwd() / "checkpoints"

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
            device=device,
            checkpoint_dir=checkpoint_dir,
            rank=rank,
            world_size=world_size,
            start_epoch=0,
            best_val_acc=0.0,
        )

    @classmethod
    def from_checkpoint_dir(
        cls,
        run_directory: Path,
        rank: int,
        world_size: int,
        local_rank: int,
        epoch: Optional[int] = None,
    ) -> "TrainState":
        """
        Create a TrainState from an existing run directory.
        Useful for resuming training or loading pretrained models for inference.

        Loads the config from the run directory's saved Hydra config,
        making this method fully self-contained.

        Args:
            run_directory: Path to the run directory (e.g., "output/runs/my_run")
            rank: Process rank
            world_size: Total number of processes
            local_rank: Local rank (GPU index)
            epoch: Specific epoch to load, or None to load best_model.pt

        Returns:
            TrainState loaded from checkpoint with original config
        """
        # Load the config from the run directory
        config_path = run_directory / ".hydra" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found in run directory: {config_path}\n"
                f"Make sure you're pointing to a valid run directory with Hydra configs."
            )

        print_rank_0(f"Loading config from: {config_path}", rank)
        config = OmegaConf.load(config_path)

        # Determine checkpoint directory from run directory
        checkpoint_dir = run_directory / "checkpoints"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

        # Create a fresh TrainState from the loaded config
        state = cls.from_config(
            config, rank, world_size, local_rank, checkpoint_dir=checkpoint_dir
        )

        # Determine checkpoint path

        if epoch is None:
            # Load best model
            checkpoint_path = checkpoint_dir / "best_model.pt"
            checkpoint_name = "best model"
        else:
            # Load specific epoch
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            checkpoint_name = f"epoch {epoch}"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print_rank_0(
            f"Loading checkpoint from {checkpoint_name}: {checkpoint_path}", rank
        )

        # Load checkpoint
        checkpoint_data = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=state.model,
            optimizer=state.optimizer,
            scheduler=state.scheduler,
            device=str(state.device),
        )

        # Update training state
        state.start_epoch = checkpoint_data["epoch"]
        state.best_val_acc = checkpoint_data.get("best_metric", 0.0) or 0.0

        print_rank_0(
            f"Loaded checkpoint from epoch {state.start_epoch}, "
            f"best val acc: {state.best_val_acc:.2f}%",
            rank,
        )

        return state
