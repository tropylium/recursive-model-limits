import os

import torch
import torch.distributed as dist


def setup_distributed():
    """
    Initialize the distributed environment.
    Automatically detects if running under torchrun via environment variables.
    
    Returns:
        tuple: (rank, world_size, local_rank) or (0, 1, 0) if not distributed
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        # Initialize the process group
        dist.init_process_group(backend="nccl")
        
        # Set the device for this process
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        # Not running in distributed mode
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return 0, 1, 0


def cleanup_distributed():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int = None) -> bool:
    """Check if this is the main process (rank 0)."""
    if rank is not None:
        return rank == 0
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_rank() -> int:
    """Get the rank of the current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get the world size (total number of processes)."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduce a tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        average: If True, compute the average. Otherwise, compute the sum.
        
    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt = rt / get_world_size()
    return rt


def broadcast_string(s: str, src: int = 0) -> str:
    """
    Broadcast a string from source rank to all other ranks.

    Args:
        s: String to broadcast (only used on source rank)
        src: Source rank (default: 0)

    Returns:
        Broadcasted string
    """
    if not dist.is_initialized():
        return s

    rank = get_rank()

    # First broadcast the length
    if rank == src:
        length = len(s.encode("utf-8"))
    else:
        length = 0

    length_tensor = torch.tensor(length, dtype=torch.long, device="cuda")
    dist.broadcast(length_tensor, src=src)
    length = length_tensor.item()

    # Then broadcast the string content
    if rank == src:
        byte_tensor = torch.tensor(
            list(s.encode("utf-8")), dtype=torch.uint8, device="cuda"
        )
    else:
        byte_tensor = torch.zeros(length, dtype=torch.uint8, device="cuda")

    dist.broadcast(byte_tensor, src=src)

    # Convert back to string
    result = bytes(byte_tensor.cpu().numpy()).decode("utf-8")
    return result
