import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables BEFORE importing modules that use them
load_dotenv()

from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from src.train import train
from src.train_state import TrainState
from src.utils.distributed import (
    barrier,
    broadcast_string,
    cleanup_distributed,
    setup_distributed,
)
from src.utils.logging import finish_wandb, init_wandb


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    """
    Main training entry point with Hydra configuration.

    Args:
        config: Hydra configuration object
    """
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    # Generate run_name on rank 0 only, then broadcast to all ranks
    # This ensures all processes use the same output directory
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{config.arch.model}_{config.dataset.name}_{timestamp}"
    else:
        run_name = ""

    # Broadcast run_name from rank 0 to all other ranks
    if world_size > 1:
        run_name = broadcast_string(run_name, src=0)
        barrier()

    # Update config with synchronized run_name
    with open_dict(config):
        config.run_name = run_name

    # Create output directory structure (only on rank 0)
    output_dir = Path("output") / "runs" / run_name
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config to output directory
        config_save_path = output_dir / "config.yaml"
        with open(config_save_path, "w") as f:
            f.write(OmegaConf.to_yaml(config))

        print("=" * 80)
        print(f"Run name: {run_name}")
        print(f"Output directory: {output_dir}")
        print("=" * 80)
        print("Configuration:")
        print(OmegaConf.to_yaml(config))
        print("=" * 80)

    # Synchronize after directory creation
    if world_size > 1:
        barrier()

    # Initialize wandb (only on rank 0)
    init_wandb(config, run_name=run_name, rank=rank)

    # Determine checkpoint directory (same for all ranks)
    checkpoint_dir = output_dir / "checkpoints"

    # Create TrainState - either from scratch or from checkpoint
    if config.training.get("resume_run_dir") is not None:
        # Resume from run directory (loads config from the run directory)
        train_state = TrainState.from_checkpoint_dir(
            run_directory=Path(config.training.resume_run_dir),
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            epoch=config.training.get("resume_epoch"),
        )
    elif config.training.get("resume_from") is not None:
        # Backwards compatibility: resume from explicit checkpoint path
        train_state = TrainState.from_config(
            config, rank, world_size, local_rank, checkpoint_dir=checkpoint_dir
        )
        train_state = train_state.load_from_checkpoint(
            Path(config.training.resume_from)
        )
    else:
        # New training run
        train_state = TrainState.from_config(
            config, rank, world_size, local_rank, checkpoint_dir=checkpoint_dir
        )

    # Start training
    train(train_state)

    # Cleanup
    finish_wandb(rank=rank)
    cleanup_distributed()


if __name__ == "__main__":
    main()
