from src.train_state import TrainState
from src.trainer import Trainer
from src.utils.logging import print_rank_0


def train(state: TrainState) -> None:
    """
    Main training loop.

    Args:
        state: TrainState containing all training components and configuration
    """
    # Initialize trainer from TrainState
    trainer = Trainer.from_train_state(state)

    # Store trainer in state for potential future use
    state.trainer = trainer

    print_rank_0("=" * 80, state.rank)
    if state.start_epoch > 0:
        print_rank_0(
            f"Resuming training from epoch {state.start_epoch + 1}", state.rank
        )
        print_rank_0(
            f"Training for {state.config.training.epochs - state.start_epoch} more epochs",
            state.rank,
        )
    else:
        print_rank_0(
            f"Starting training for {state.config.training.epochs} epochs", state.rank
        )
    print_rank_0(f"Batch size per GPU: {state.config.training.batch_size}", state.rank)
    print_rank_0(
        f"Total batch size: {state.config.training.batch_size * state.world_size}",
        state.rank,
    )
    print_rank_0(f"Using AMP: {state.config.training.use_amp}", state.rank)
    print_rank_0(
        f"Validation every {state.config.training.checkpoint_every} epochs", state.rank
    )
    print_rank_0("=" * 80, state.rank)

    # Training loop
    for epoch in range(state.start_epoch, state.config.training.epochs):
        # Train for one epoch
        train_metrics = trainer.train_epoch(state.train_loader, epoch)

        # Check if this is a checkpoint epoch or the last epoch
        is_checkpoint_epoch = (epoch + 1) % state.config.training.checkpoint_every == 0
        is_last_epoch = epoch == state.config.training.epochs - 1
        should_validate = is_checkpoint_epoch or is_last_epoch

        # Validate only on checkpoint epochs
        if should_validate:
            val_metrics = trainer.validate(state.val_loader, epoch)
            metrics = {**train_metrics, **val_metrics}

            # Print metrics with validation
            print_rank_0(
                f"Epoch {epoch + 1}/{state.config.training.epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.2f}%",
                state.rank,
            )
        else:
            metrics = train_metrics

            # Print metrics without validation
            print_rank_0(
                f"Epoch {epoch + 1}/{state.config.training.epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_accuracy']:.2f}%",
                state.rank,
            )

        # Log to wandb
        trainer.log_metrics_to_wandb(metrics, epoch)

        # Step scheduler
        trainer.step_scheduler()

        # Save checkpoint if needed (only on checkpoint epochs)
        if should_validate:
            trainer.save_checkpoint_if_needed(epoch, metrics)

    print_rank_0("=" * 80, state.rank)
    print_rank_0(
        f"Training complete! Best Val Acc: {trainer.best_val_acc:.2f}%", state.rank
    )
    print_rank_0("=" * 80, state.rank)
