import os
import json
import torch
import pandas as pd
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union, Any

from data.datasets.base import MultiModalDataset


class HatefulMemesDataset(MultiModalDataset):
    """
    Dataset for the Hateful Memes challenge.
    
    This dataset loads meme images and their accompanying text, classifying them as hateful or not.
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        transform=None,
        text_transform=None,
        image_key: str = 'img',
        text_key: str = 'text',
        label_key: str = 'label'
    ):
        """
        Initialize the hateful memes dataset.
        
        Args:
            data_path: Path to the dataset root directory
            split: Dataset split ('train', 'val', or 'test')
            transform: Image transformation pipeline
            text_transform: Text transformation pipeline
            image_key: Key for image in the dataset
            text_key: Key for text in the dataset
            label_key: Key for label in the dataset
        """
        super().__init__(modalities=['image', 'text'])
        
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.text_transform = text_transform
        self.image_key = image_key
        self.text_key = text_key
        self.label_key = label_key
        
        # Load dataset
        annotation_file = os.path.join(data_path, f"{split}.jsonl")
        self.data = []
        
        # Load jsonl file
        with open(annotation_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
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
        item = self.data[index]
        
        # Load image
        img_path = os.path.join(self.data_path, item[self.image_key])
        image = Image.open(img_path).convert('RGB')
        
        # Apply image transformation if available
        if self.transform:
            image = self.transform(image)
        
        # Get text
        text = item[self.text_key]
        
        # Apply text transformation if available
        if self.text_transform:
            text = self.text_transform(text)
        
        # Get label
        if self.label_key in item:
            label = item[self.label_key]
            label = torch.tensor(label, dtype=torch.long)
        else:
            # For test set without labels
            label = torch.tensor(-1, dtype=torch.long)
        
        return {
            'image': image,
            'text': text,
            'target': label,
            'id': item.get('id', index)
        }