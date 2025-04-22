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
    pin_memory: bool = True,
    persistent_workers: bool = True,
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
    else:
        sampler = None
    
    # Handle multiprocessing for Windows
    if num_workers > 0 and persistent_workers and torch.get_num_threads() > 1:
        persistent_workers = True
    else:
        persistent_workers = False
    
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