import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables BEFORE importing modules that use them
load_dotenv()

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.datasets import dataset_factory
from src.models.loader import load_model_class
from src.train import train
from src.utils.distributed import cleanup_distributed, setup_distributed
from src.utils.logging import finish_wandb, init_wandb, print_rank_0


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    """
    Main training entry point with Hydra configuration.

    Args:
        config: Hydra configuration object
    """
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Print config on rank 0
    if rank == 0:
        print("=" * 80)
        print("Configuration:")
        print(OmegaConf.to_yaml(config))
        print("=" * 80)

    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Initialize wandb (only on rank 0)
    run_name = config.get("run_name", None)
    init_wandb(config, run_name=run_name, rank=rank)

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
        # Don't shuffle when using DistributedSampler
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
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
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

    # Setup checkpoint directory (inside Hydra's output directory)
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    checkpoint_dir = hydra_output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Start training
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        rank=rank,
        world_size=world_size,
    )

    # Cleanup
    finish_wandb(rank=rank)
    cleanup_distributed()


if __name__ == "__main__":
    main()
