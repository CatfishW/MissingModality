import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union


class MultiModalDataset(Dataset, ABC):
    """
    Base class for multi-modal datasets.
    
    This abstract class provides a structure for datasets with multiple modalities.
    Implementations should override the __getitem__ and __len__ methods.
    """
    
    def __init__(self, modalities: List[str]):
        """
        Initialize the multi-modal dataset.
        
        Args:
            modalities (List[str]): List of modality names
        """
        self.modalities = modalities
    
    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get an item from the dataset.
        
        Args:
            index (int): Index of the item
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary with modality names as keys and corresponding
                tensors as values, plus 'target' key for the ground truth
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        pass


class MissingModalityDataset(Dataset):
    """
    Wrapper for datasets to simulate missing modalities during testing.
    
    This class wraps a multi-modal dataset and simulates missing modalities
    by dropping specified modalities from the data.
    """
    
    def __init__(
        self, 
        dataset: MultiModalDataset,
        missing_modalities: Optional[List[str]] = None,
        missing_prob: float = 0.0,
        random_missing: bool = False
    ):
        """
        Initialize the missing modality dataset wrapper.
        
        Args:
            dataset (MultiModalDataset): Base multi-modal dataset
            missing_modalities (Optional[List[str]]): List of modalities to drop
            missing_prob (float): Probability of dropping a modality (used when random_missing=True)
            random_missing (bool): Whether to randomly drop modalities
        """
        self.dataset = dataset
        self.modalities = dataset.modalities
        self.missing_modalities = missing_modalities or []
        self.missing_prob = missing_prob
        self.random_missing = random_missing
    
    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, None]]:
        """
        Get an item from the dataset with potentially missing modalities.
        
        Args:
            index (int): Index of the item
        
        Returns:
            Dict[str, Union[torch.Tensor, None]]: Dictionary with modality data and targets,
                with None for missing modalities
        """
        data = self.dataset[index]
        
        # Create a mask for missing modalities
        if self.random_missing:
            # Randomly drop modalities based on probability
            for modality in self.modalities:
                if modality in data and torch.rand(1).item() < self.missing_prob:
                    data[modality] = None
        else:
            # Drop specified modalities
            for modality in self.missing_modalities:
                if modality in data:
                    data[modality] = None
        
        # Add missing modality information
        missing_mask = {f"{modality}_missing": (data[modality] is None) 
                       for modality in self.modalities if modality in data}
        data.update(missing_mask)
        
        return data
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.dataset)