import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any


class MissingModalityModel(nn.Module):
    """
    Base class for models that can handle missing modalities.
    
    This class provides a framework for creating models that can gracefully
    handle missing modalities during inference.
    """
    
    def __init__(self, modalities: List[str]):
        """
        Initialize the missing modality model.
        
        Args:
            modalities (List[str]): List of modality names
        """
        super().__init__()
        self.modalities = modalities
        
    def forward(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            inputs (Dict[str, Any]): Input dictionary with modality data
                Each modality should be a key in the dictionary with the 
                corresponding tensor as the value. Missing modalities can
                be represented as None.
        
        Returns:
            Dict[str, torch.Tensor]: Output dictionary with model predictions
        """
        # Extract modalities and check which are missing
        modality_features = {}
        missing_modalities = []
        
        for modality in self.modalities:
            if modality in inputs and inputs[modality] is not None:
                # Extract features for available modalities
                modality_features[modality] = self.extract_features(modality, inputs[modality])
            else:
                missing_modalities.append(modality)
        
        # Handle missing modalities
        if missing_modalities:
            modality_features = self.handle_missing_modalities(
                modality_features, missing_modalities, inputs
            )
        
        # Fuse features from different modalities
        fused_features = self.fuse_modalities(modality_features, inputs)
        
        # Get predictions from fused features
        outputs = self.get_predictions(fused_features, inputs)
        
        return outputs
    
    def extract_features(self, modality: str, inputs: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a single modality.
        
        Args:
            modality (str): Name of the modality
            inputs (torch.Tensor): Input tensor for the modality
        
        Returns:
            torch.Tensor: Features extracted from the modality
        """
        raise NotImplementedError("Subclasses must implement extract_features")
    
    def handle_missing_modalities(
        self, 
        modality_features: Dict[str, torch.Tensor],
        missing_modalities: List[str],
        inputs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Handle missing modalities by synthesizing features or using defaults.
        
        Args:
            modality_features (Dict[str, torch.Tensor]): Features from available modalities
            missing_modalities (List[str]): List of missing modality names
            inputs (Dict[str, Any]): Original input dictionary
        
        Returns:
            Dict[str, torch.Tensor]: Updated modality features
        """
        return modality_features  # Default implementation does nothing
    
    def fuse_modalities(
        self, 
        modality_features: Dict[str, torch.Tensor],
        inputs: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Fuse features from different modalities.
        
        Args:
            modality_features (Dict[str, torch.Tensor]): Features from all modalities
            inputs (Dict[str, Any]): Original input dictionary
        
        Returns:
            torch.Tensor: Fused features
        """
        raise NotImplementedError("Subclasses must implement fuse_modalities")
    
    def get_predictions(
        self, 
        fused_features: torch.Tensor,
        inputs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Get predictions from fused features.
        
        Args:
            fused_features (torch.Tensor): Fused features from all modalities
            inputs (Dict[str, Any]): Original input dictionary
        
        Returns:
            Dict[str, torch.Tensor]: Output dictionary with model predictions
        """
        raise NotImplementedError("Subclasses must implement get_predictions")