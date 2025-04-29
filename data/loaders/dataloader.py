import torch
from torch.utils.data import DataLoader, DistributedSampler
from typing import Any, Callable, Dict, Optional, Union

from data.datasets.base import MultiModalDataset, MissingModalityDataset


def create_dataloader(
    dataset: Union[MultiModalDataset, MissingModalityDataset],
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = False,
    distributed: bool = False,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    **kwargs
) -> DataLoader:
    """
    Create a PyTorch DataLoader with support for distributed training.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size for the dataloader
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        distributed: Whether to use distributed sampling
        collate_fn: Custom collate function
        pin_memory: Whether to pin memory for faster GPU transfer
        persistent_workers: Whether to keep worker processes alive between iterations
        **kwargs: Additional arguments to pass to DataLoader
    
    Returns:
        DataLoader: PyTorch DataLoader instance
    """
    if distributed and torch.distributed.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Shuffle is handled by sampler
        print(f"Using DistributedSampler with {torch.distributed.get_world_size()} processes.")
    else:
        sampler = None
        print("Using default sampler.")
    
    # Handle multiprocessing settings for better stability, especially on Windows
    if torch.get_num_threads() <= 1:
        # If system is running with few CPU threads, disable multiprocessing
        num_workers = 0
        persistent_workers = False
    
    # On Windows, sometimes persistent_workers can cause issues
    if num_workers == 0:
        persistent_workers = False
    elif persistent_workers and num_workers > 0 and (torch.get_num_threads() > 1):
        # Only use persistent workers when we have sufficient resources
        persistent_workers = True
    else:
        persistent_workers = False
    
    # Use a more conservative approach for Windows
    import platform
    if platform.system() == 'Windows':
        # Windows has more issues with worker processes
        num_workers = min(num_workers, 2)  # Limit workers on Windows
        persistent_workers = False  # Disable persistent workers on Windows
    
    try:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            **kwargs
        )
    except Exception as e:
        print(f"Error creating DataLoader: {e}")
        print("Falling back to safer DataLoader configuration...")
        # Fall back to a safer configuration
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,  # Disable multiprocessing
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=False,  # Disable pin memory
            persistent_workers=False,  # Disable persistent workers
            **kwargs
        )


def default_collate_fn(batch: list) -> Dict[str, Any]:
    """
    Default collate function for multi-modal data.
    
    This collate function handles None values for missing modalities.
    
    Args:
        batch: List of samples from the dataset
    
    Returns:
        Dict: Collated batch with modality data and targets
    """
    if not batch:
        return {}
    
    result = {}
    keys = batch[0].keys()
    
    for key in keys:
        values = [sample[key] for sample in batch]
        
        # Skip None values (missing modalities)
        valid_values = [v for v in values if v is not None]
        
        if not valid_values:
            # All values are None for this modality
            result[key] = None
        else:
            # Stack valid values
            if isinstance(valid_values[0], torch.Tensor):
                result[key] = torch.stack(valid_values)
            elif isinstance(valid_values[0], (bool, int, float)):
                result[key] = torch.tensor(valid_values)
            else:
                result[key] = valid_values
    
    return result