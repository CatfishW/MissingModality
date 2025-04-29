import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel, RobertaModel
from typing import Dict, List, Any, Optional

from models.missing.base_model import MissingModalityModel
from models.fusion.fusion_modules import EarlyFusion, LateFusion, AttentionFusion


class HatefulMemesClassifier(MissingModalityModel):
    """
    Multi-modal classifier for hateful memes detection.
    
    This model processes both image and text modalities and classifies
    whether a meme is hateful or not.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        image_encoder: str = "resnet50",
        text_encoder: str = "bert-base-uncased",
        fusion_type: str = "attention",
        fusion_dim: int = 512,
        dropout: float = 0.2,
        pretrained: bool = True,
        handling_missing: Dict[str, Any] = {"strategy": "zero"}
    ):
        """
        Initialize the hateful memes classifier.
        
        Args:
            num_classes: Number of classes (typically 2 for binary classification)
            image_encoder: Image backbone model name
            text_encoder: Text backbone model name
            fusion_type: Type of fusion ('early', 'late', or 'attention')
            fusion_dim: Dimension of fusion features
            dropout: Dropout probability
            pretrained: Whether to use pretrained backbones
            handling_missing: Strategy for handling missing modalities
        """
        super().__init__(modalities=["image", "text"])
        
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        self.fusion_dim = fusion_dim
        self.handling_missing = handling_missing
        
        # Image encoder
        if image_encoder == "resnet18":
            self.image_backbone = models.resnet18(pretrained=pretrained)
            image_dim = 512
        elif image_encoder == "resnet50":
            self.image_backbone = models.resnet50(pretrained=pretrained)
            image_dim = 2048
        elif image_encoder == "resnet101":
            self.image_backbone = models.resnet101(pretrained=pretrained)
            image_dim = 2048
        else:
            raise ValueError(f"Unsupported image encoder: {image_encoder}")
        
        # Remove final classification layer
        self.image_backbone = nn.Sequential(*list(self.image_backbone.children())[:-1])
        
        # Image projection layer
        self.image_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(image_dim, fusion_dim),
            nn.ReLU()
        )
        
        # Text encoder
        if "roberta" in text_encoder:
            self.text_backbone = RobertaModel.from_pretrained(text_encoder)
            text_dim = self.text_backbone.config.hidden_size
        else:
            self.text_backbone = BertModel.from_pretrained(text_encoder)
            text_dim = self.text_backbone.config.hidden_size
        
        # Text projection layer
        self.text_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(text_dim, fusion_dim),
            nn.ReLU()
        )
        
        # Fusion module
        feature_dims = {
            "image": fusion_dim,
            "text": fusion_dim
        }
        
        if fusion_type == "early":
            self.fusion = EarlyFusion(feature_dims, output_dim=fusion_dim)
        elif fusion_type == "late":
            self.fusion = LateFusion(feature_dims, output_dim=num_classes, fusion_type="weighted_sum")
        elif fusion_type == "attention":
            self.fusion = AttentionFusion(feature_dims, output_dim=fusion_dim)
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        
        # Classification head (only for early and attention fusion)
        if fusion_type != "late":
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(fusion_dim, num_classes)
            )
        
        # Initialize defaults for handling missing modalities
        self.register_buffer("image_default", torch.zeros(1, fusion_dim))
        self.register_buffer("text_default", torch.zeros(1, fusion_dim))
        
        # If strategy is "learned", create learnable parameters
        strategy = handling_missing.get("strategy", "zero")
        if strategy == "learned":
            self.image_default = nn.Parameter(torch.randn(1, fusion_dim) * 0.02)
            self.text_default = nn.Parameter(torch.randn(1, fusion_dim) * 0.02)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
    def extract_features(self, modality: str, inputs: Any) -> torch.Tensor: # Changed type hint for inputs
        """
        Extract features from a single modality.
        
        Args:
            modality: Name of the modality ('image' or 'text')
            inputs: Input tensor or list/dict for the modality
        
        Returns:
            Features extracted from the modality
        """
        if modality == "image":
            # Process image through backbone
            features = self.image_backbone(inputs)
            features = features.flatten(start_dim=1)
            return self.image_projection(features)
        
        elif modality == "text":
            # Process text through backbone
            # Convert list of dicts to batched tensors
            if isinstance(inputs, list):
                # Extract tensors from list of dictionaries
                input_ids = torch.stack([item["input_ids"] for item in inputs]).to(self.device)
                attention_mask = torch.stack([item["attention_mask"] for item in inputs]).to(self.device)
                
                # token_type_ids might not be present for RoBERTa models
                token_type_ids = None
                if "token_type_ids" in inputs[0]:
                    token_type_ids = torch.stack([item["token_type_ids"] for item in inputs]).to(self.device)
            else:
                # Handle case where inputs might already be batched
                input_ids = inputs.get("input_ids")
                token_type_ids = inputs.get("token_type_ids")
                attention_mask = inputs.get("attention_mask")
            # Pass inputs to the text backbone
            text_outputs = self.text_backbone(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            
            # Get the pooled output (CLS token representation)
            pooled_output = text_outputs.pooler_output
            
            # Project to fusion dimension
            return self.text_projection(pooled_output)
        
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    def handle_missing_modalities(
        self, 
        modality_features: Dict[str, torch.Tensor],
        missing_modalities: List[str],
        inputs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Handle missing modalities based on specified strategy.
        
        Args:
            modality_features: Features from available modalities
            missing_modalities: List of missing modality names
            inputs: Original input dictionary
        
        Returns:
            Updated modality features with missing modalities filled in
        """
        strategy = self.handling_missing.get("strategy", "zero")
        batch_size = next(iter(modality_features.values())).size(0) if modality_features else 1
        
        for modality in missing_modalities:
            if modality == "image":
                if strategy == "zero" or strategy == "learned":
                    # Use zero tensor or learned parameter
                    modality_features[modality] = self.image_default.expand(batch_size, -1)
                elif strategy == "mean" and "text" in modality_features:
                    # Use mean of other modalities
                    modality_features[modality] = modality_features["text"].clone()
            
            elif modality == "text":
                if strategy == "zero" or strategy == "learned":
                    # Use zero tensor or learned parameter
                    modality_features[modality] = self.text_default.expand(batch_size, -1)
                elif strategy == "mean" and "image" in modality_features:
                    # Use mean of other modalities
                    modality_features[modality] = modality_features["image"].clone()
        
        return modality_features
    
    def fuse_modalities(
        self, 
        modality_features: Dict[str, torch.Tensor],
        inputs: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Fuse features from different modalities.
        
        Args:
            modality_features: Features from all modalities
            inputs: Original input dictionary
        
        Returns:
            Fused features
        """
        return self.fusion(modality_features)
    
    def get_predictions(
        self, 
        fused_features: torch.Tensor,
        inputs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Get predictions from fused features.
        
        Args:
            fused_features: Fused features from all modalities
            inputs: Original input dictionary
        
        Returns:
            Dictionary with model predictions
        """
        if self.fusion_type == "late":
            # Late fusion returns logits directly
            logits = fused_features
        else:
            # Apply classifier for early and attention fusion
            logits = self.classifier(fused_features)
        
        return {
            "logits": logits,
            "preds": torch.argmax(logits, dim=1)
        }