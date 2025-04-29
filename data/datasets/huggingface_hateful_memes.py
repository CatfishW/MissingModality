import os
import torch
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Any
from datasets import load_dataset

from data.datasets.base import MultiModalDataset


class HuggingFaceHatefulMemesDataset(MultiModalDataset):
    """
    Dataset for the Hateful Memes challenge loaded from Hugging Face datasets.
    
    This dataset loads meme images and their accompanying text directly from
    the Hugging Face datasets library, classifying them as hateful or not.
    """
    
    def __init__(
        self,
        split: str = 'train',
        transform=None,
        text_transform=None,
        dataset_name: str = "neuralcatcher/hateful_memes",
        cache_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
    ):
        """
        Initialize the Hugging Face hateful memes dataset.
        
        Args:
            split: Dataset split ('train', 'validation', or 'test')
            transform: Image transformation pipeline
            text_transform: Text transformation pipeline
            dataset_name: Name of the Hugging Face dataset
            cache_dir: Optional directory to cache the downloaded dataset
            data_dir: Optional base directory for images
        """
        super().__init__(modalities=['image', 'text'])
        
        # Map split name to Hugging Face split name
        hf_split = split
        if split == 'val':
            hf_split = 'validation'
        
        # Load dataset from Hugging Face
        self.dataset = load_dataset(dataset_name, split=hf_split, cache_dir=cache_dir)
        
        self.transform = transform
        self.text_transform = text_transform
        self.data_dir = data_dir
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            index: Index of the sample
        
        Returns:
            Dictionary containing:
                'image': Image tensor
                'text': Text tensor
                'target': Binary label (1 for hateful, 0 for not hateful)
        """
        item = self.dataset[index]
        
        # Load image - Hugging Face might return the image path or a PIL Image
        image = item['img']
        
        # If image is a string (path), load it as a PIL Image
        if isinstance(image, str):
            # Try different possible image paths
            image_paths = [
                image,  # Original path
                os.path.join(self.data_dir, image) if self.data_dir else None,  # With data_dir prefix
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", image)  # Relative to project root
            ]
            
            image_loaded = False
            for img_path in image_paths:
                if img_path and os.path.exists(img_path):
                    try:
                        image = Image.open(img_path).convert('RGB')
                        image_loaded = True
                        break
                    except Exception:
                        continue
            
            if not image_loaded:
                # Create a dummy image (gray) if image can't be loaded
                image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Apply image transformation if available
        if self.transform:
            image = self.transform(image)
        
        # Get text
        text = item['text']
        
        # Apply text transformation if available
        if self.text_transform:
            text_inputs = self.text_transform(text)
            
            # Ensure all tensor values in the text_inputs dict are on the same device
            if isinstance(text_inputs, dict):
                for key, value in text_inputs.items():
                    if isinstance(value, list):
                        # Convert lists to tensors
                        text_inputs[key] = torch.tensor(value)
        else:
            text_inputs = text
        
        # Get label
        if 'label' in item:
            label = item['label']
            label = torch.tensor(label, dtype=torch.long)
        else:
            # For test set without labels
            label = torch.tensor(-1, dtype=torch.long)
        
        return {
            'image': image,
            'text': text_inputs,
            'target': label,
            'id': item.get('id', index)
        }