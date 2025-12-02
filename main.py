import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables BEFORE importing modules that use them
load_dotenv()

import hydra
from omegaconf import DictConfig, OmegaConf

from src.train import train
from src.train_state import TrainState
from src.utils.distributed import cleanup_distributed, setup_distributed
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

    # Print config on rank 0
    if rank == 0:
        print("=" * 80)
        print("Configuration:")
        print(OmegaConf.to_yaml(config))
        print("=" * 80)

    # Initialize wandb (only on rank 0)
    run_name = config.get("run_name", None)
    init_wandb(config, run_name=run_name, rank=rank)

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
        train_state = TrainState.from_config(config, rank, world_size, local_rank)
        train_state = train_state.load_from_checkpoint(
            Path(config.training.resume_from)
        )
    else:
        # New training run
        train_state = TrainState.from_config(config, rank, world_size, local_rank)

    # Start training
    train(train_state)

    # Cleanup
    finish_wandb(rank=rank)
    cleanup_distributed()


if __name__ == "__main__":
    main()
