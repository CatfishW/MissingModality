import torch
import torchvision.transforms as T
from typing import List, Union, Dict, Any
from transformers import BertTokenizer, RobertaTokenizer

class TextTransform:
    """A class-based text transform that can be pickled."""
    
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __call__(self, text: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Ensure all values are tensors (not lists) and squeeze to remove batch dimension
        result = {}
        for k, v in encoded.items():
            if isinstance(v, list):
                result[k] = torch.tensor(v, device=self.device)
            else:
                # Already a tensor, just squeeze the batch dimension
                result[k] = v.squeeze(0)
        
        return result

class HatefulMemesTransforms:
    """
    Transformations for the Hateful Memes dataset.
    
    This class provides transformations for both image and text modalities
    used in the Hateful Memes dataset.
    """
    
    @staticmethod
    def get_image_transform(split: str = "train", image_size: int = 224, normalize: bool = True):
        """
        Get image transformation pipeline.
        
        Args:
            split: Dataset split ('train', 'val', or 'test')
            image_size: Target image size
            normalize: Whether to normalize the image with ImageNet stats
        
        Returns:
            Torchvision transformation pipeline for images
        """
        if split == "train":
            transform = T.Compose([
                T.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ToTensor()
            ])
        else:
            transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor()
            ])
        
        # Add normalization if needed
        if normalize:
            transform.transforms.append(
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )
        
        return transform
    
    @staticmethod
    def get_text_transform(model_name: str = "bert-base-uncased", max_length: int = 128):
        """
        Get text transformation pipeline using a pretrained tokenizer.
        
        Args:
            model_name: Name of the pretrained model to use for tokenization
            max_length: Maximum sequence length
        
        Returns:
            Text transformation function that tokenizes input text
        """
        if "roberta" in model_name:
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
        else:
            tokenizer = BertTokenizer.from_pretrained(model_name)
        
        return TextTransform(tokenizer, max_length)