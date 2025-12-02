import os
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

import wandb


def init_wandb(
    config: DictConfig,
    project_name: Optional[str] = None,
    run_name: Optional[str] = None,
    rank: int = 0
) -> None:
    """
    Initialize Weights & Biases logging (only on rank 0).
    
    Args:
        config: Hydra config object
        project_name: WandB project name (defaults to env var WANDB_PROJECT_NAME)
        run_name: Run name for this experiment
        rank: Process rank (only rank 0 will initialize wandb)
    """
    if rank != 0:
        return
    
    if project_name is None:
        project_name = os.environ.get("WANDB_PROJECT_NAME", "recursive-model-limits")
    
    # Convert OmegaConf to dict for wandb
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Create settings to disable system metrics logging
    wandb_settings = wandb.Settings(
        _disable_stats=True,  # Disable system stats (GPU, CPU, memory monitoring)
        # _disable_meta=True,   # Disable git metadata collection
    )
    
    wandb.init(
        project=project_name,
        name=run_name,
        config=config_dict,
        settings=wandb_settings,
    )


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None, rank: int = 0) -> None:
    """
    Log metrics to wandb (only on rank 0).
    
    Args:
        metrics: Dictionary of metric names and values
        step: Training step/epoch number
        rank: Process rank (only rank 0 will log)
    """
    if rank != 0:
        return
    
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_wandb(rank: int = 0) -> None:
    """
    Finish wandb run (only on rank 0).
    
    Args:
        rank: Process rank (only rank 0 will finish)
    """
    if rank != 0:
        return
    
    if wandb.run is not None:
        wandb.finish()


def print_rank_0(message: str, rank: int = 0) -> None:
    """
    Print message only on rank 0.
    
    Args:
        message: Message to print
        rank: Process rank
    """
    if rank == 0:
        print(message)

