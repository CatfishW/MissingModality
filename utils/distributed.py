import os
import random
import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed(backend='nccl', port=None):
    """
    Initialize distributed training.
    
    Args:
        backend (str): Distributed backend ('nccl' for GPU, 'gloo' for CPU)
        port (int, optional): Port for distributed communication
    
    Returns:
        tuple: (rank, world_size) - current process rank and total number of processes
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = 0
        world_size = 1
    
    if world_size > 1:
        if port is None:
            port = find_free_port()
        
        dist.init_process_group(
            backend=backend,
            init_method=f'tcp://127.0.0.1:{port}',
            world_size=world_size,
            rank=rank
        )
    
    return rank, world_size


def find_free_port():
    """Find a free port on the machine."""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def all_gather(tensor):
    """
    Gather tensors from all processes and concatenate them.
    
    Args:
        tensor (torch.Tensor): Tensor to gather
    
    Returns:
        torch.Tensor: Concatenated tensor from all processes
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    
    # Gather tensors from all GPUs
    output = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(output, tensor)
    
    return torch.cat(output, dim=0)


def reduce_tensor(tensor, average=True):
    """
    Reduce tensor across all processes.
    
    Args:
        tensor (torch.Tensor): Tensor to reduce
        average (bool): Whether to average or sum the tensor
    
    Returns:
        torch.Tensor: Reduced tensor
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor
    
    # Reduce tensors from all GPUs
    tensor = tensor.clone()
    dist.all_reduce(tensor)
    
    if average:
        tensor /= world_size
    
    return tensor


def is_main_process():
    """Check if current process is the main process."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0