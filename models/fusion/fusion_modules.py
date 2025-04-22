import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any


class EarlyFusion(nn.Module):
    """
    Early Fusion module: concatenate features from different modalities directly.
    """
    
    def __init__(self, feature_dims: Dict[str, int], output_dim: int = None):
        """
        Initialize the early fusion module.
        
        Args:
            feature_dims (Dict[str, int]): Dictionary mapping modality names to feature dimensions
            output_dim (int, optional): Output dimension after fusion. If None, no projection is applied.
        """
        super().__init__()
        self.feature_dims = feature_dims
        
        # Calculate total dimension after concatenation
        self.total_dim = sum(feature_dims.values())
        
        # Optional projection layer
        if output_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(self.total_dim, output_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.projection = nn.Identity()
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform early fusion by concatenating features.
        
        Args:
            modality_features (Dict[str, torch.Tensor]): Dictionary of features for each modality
                with shape [batch_size, feature_dim]
        
        Returns:
            torch.Tensor: Fused features with shape [batch_size, output_dim] or [batch_size, total_dim]
        """
        # Check if all expected modalities are available
        features_list = []
        
        for modality in self.feature_dims.keys():
            if modality in modality_features:
                features_list.append(modality_features[modality])
        
        # Concatenate features along feature dimension
        if len(features_list) > 1:
            fused = torch.cat(features_list, dim=1)
        elif len(features_list) == 1:
            fused = features_list[0]
        else:
            raise ValueError("No modality features available for fusion")
        
        # Apply projection if specified
        return self.projection(fused)


class LateFusion(nn.Module):
    """
    Late Fusion module: process each modality separately and combine predictions.
    """
    
    def __init__(
        self, 
        feature_dims: Dict[str, int], 
        output_dim: int,
        hidden_dim: int = 256,
        fusion_type: str = 'weighted_sum'
    ):
        """
        Initialize the late fusion module.
        
        Args:
            feature_dims (Dict[str, int]): Dictionary mapping modality names to feature dimensions
            output_dim (int): Output dimension for each modality branch
            hidden_dim (int): Hidden dimension for each modality branch
            fusion_type (str): Type of fusion - 'sum', 'mean', 'max', or 'weighted_sum'
        """
        super().__init__()
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        
        # Create separate branches for each modality
        self.branches = nn.ModuleDict()
        for modality, dim in feature_dims.items():
            self.branches[modality] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, output_dim)
            )
        
        # For weighted sum fusion, learn weights for each modality
        if fusion_type == 'weighted_sum':
            self.modality_weights = nn.Parameter(
                torch.ones(len(feature_dims)) / len(feature_dims)
            )
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform late fusion by combining predictions from each modality.
        
        Args:
            modality_features (Dict[str, torch.Tensor]): Dictionary of features for each modality
                with shape [batch_size, feature_dim]
        
        Returns:
            torch.Tensor: Fused predictions with shape [batch_size, output_dim]
        """
        # Process each modality separately
        modality_outputs = {}
        available_modalities = []
        
        for modality in self.feature_dims.keys():
            if modality in modality_features:
                modality_outputs[modality] = self.branches[modality](modality_features[modality])
                available_modalities.append(modality)
        
        if not modality_outputs:
            raise ValueError("No modality features available for fusion")
        
        # Combine outputs according to fusion type
        if self.fusion_type == 'sum':
            fused = sum(modality_outputs.values())
        elif self.fusion_type == 'mean':
            fused = sum(modality_outputs.values()) / len(modality_outputs)
        elif self.fusion_type == 'max':
            stacked = torch.stack(list(modality_outputs.values()), dim=0)
            fused, _ = torch.max(stacked, dim=0)
        elif self.fusion_type == 'weighted_sum':
            # Get indices of available modalities
            indices = [list(self.feature_dims.keys()).index(m) for m in available_modalities]
            weights = F.softmax(self.modality_weights[indices], dim=0)
            
            # Apply weights to each modality output
            weighted_outputs = [weights[i] * modality_outputs[mod] 
                              for i, mod in enumerate(available_modalities)]
            fused = sum(weighted_outputs)
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")
        
        return fused


class AttentionFusion(nn.Module):
    """
    Attention-based Fusion module: use cross-attention to fuse features.
    """
    
    def __init__(
        self, 
        feature_dims: Dict[str, int],
        output_dim: int = None,
        attention_dim: int = 256,
        num_heads: int = 4
    ):
        """
        Initialize the attention fusion module.
        
        Args:
            feature_dims (Dict[str, int]): Dictionary mapping modality names to feature dimensions
            output_dim (int, optional): Output dimension after fusion
            attention_dim (int): Dimension of attention layers
            num_heads (int): Number of attention heads
        """
        super().__init__()
        self.feature_dims = feature_dims
        
        # Project each modality to the same dimension
        self.projections = nn.ModuleDict()
        for modality, dim in feature_dims.items():
            self.projections[modality] = nn.Linear(dim, attention_dim)
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output projection
        if output_dim is not None:
            self.output_projection = nn.Linear(attention_dim, output_dim)
        else:
            self.output_projection = nn.Identity()
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform attention-based fusion of modality features.
        
        Args:
            modality_features (Dict[str, torch.Tensor]): Dictionary of features for each modality
                with shape [batch_size, feature_dim]
        
        Returns:
            torch.Tensor: Fused features with shape [batch_size, output_dim] or [batch_size, attention_dim]
        """
        # Project each modality to the same dimension
        projected_features = {}
        for modality, features in modality_features.items():
            if modality in self.projections:
                projected_features[modality] = self.projections[modality](features).unsqueeze(1)
        
        if not projected_features:
            raise ValueError("No modality features available for fusion")
        
        # Concatenate projected features
        features_sequence = torch.cat(list(projected_features.values()), dim=1)
        
        # Apply multi-head attention
        attn_output, _ = self.attention(
            query=features_sequence,
            key=features_sequence,
            value=features_sequence
        )
        
        # Average over the sequence dimension
        fused = attn_output.mean(dim=1)
        
        # Apply output projection
        return self.output_projection(fused)