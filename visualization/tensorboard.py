import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Union, Any


class TensorboardVisualizer:
    """
    Utility class for visualizing results in Tensorboard.
    """
    
    def __init__(self, writer: SummaryWriter):
        """
        Initialize the visualizer.
        
        Args:
            writer: TensorBoard SummaryWriter
        """
        self.writer = writer
    
    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int):
        """
        Add multiple scalars to TensorBoard.
        
        Args:
            main_tag: Group name for the scalars
            tag_scalar_dict: Dictionary of tag name to scalar value
            global_step: Global step value to record
        """
        for tag, scalar in tag_scalar_dict.items():
            self.writer.add_scalar(f"{main_tag}/{tag}", scalar, global_step)
    
    def add_image_grid(self, tag: str, images: torch.Tensor, global_step: int, normalize: bool = True):
        """
        Add a grid of images to TensorBoard.
        
        Args:
            tag: Tag for the image grid
            images: Tensor of images (N, C, H, W)
            global_step: Global step value
            normalize: Whether to normalize the image to [0, 1]
        """
        if normalize:
            images = self._normalize_images(images)
        
        self.writer.add_images(tag, images, global_step)
    
    def add_confusion_matrix(self, tag: str, confusion_matrix: torch.Tensor, class_names: List[str], global_step: int):
        """
        Add a confusion matrix plot to TensorBoard.
        
        Args:
            tag: Tag for the confusion matrix
            confusion_matrix: Confusion matrix tensor
            class_names: List of class names
            global_step: Global step value
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        cax = ax.matshow(confusion_matrix.cpu().numpy())
        fig.colorbar(cax)
        
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, str(confusion_matrix[i, j].item()),
                        ha="center", va="center", color="w" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black")
        
        plt.tight_layout()
        self.writer.add_figure(tag, fig, global_step)
        plt.close(fig)
    
    def add_pr_curve(self, tag: str, labels: torch.Tensor, predictions: torch.Tensor, global_step: int, num_thresholds: int = 127):
        """
        Add precision-recall curve to TensorBoard.
        
        Args:
            tag: Tag for the PR curve
            labels: Ground truth labels (0 or 1)
            predictions: Prediction probabilities [0, 1]
            global_step: Global step value
            num_thresholds: Number of thresholds for PR curve
        """
        self.writer.add_pr_curve(
            tag,
            labels=labels,
            predictions=predictions,
            global_step=global_step,
            num_thresholds=num_thresholds
        )
    
    def add_modality_comparison(
        self, 
        tag: str, 
        full_modality_metrics: Dict[str, float],
        missing_modality_metrics: Dict[str, Dict[str, float]],
        global_step: int
    ):
        """
        Add comparison between full modality and missing modality results.
        
        Args:
            tag: Tag for the comparison
            full_modality_metrics: Metrics when all modalities are present
            missing_modality_metrics: Dict of metrics for each missing modality scenario
            global_step: Global step value
        """
        # Create a bar chart comparing performance
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        # Choose the first metric to plot (e.g., accuracy, mAP)
        metric_key = list(full_modality_metrics.keys())[0]
        
        # Organize data for plotting
        labels = ['Full Modality'] + list(missing_modality_metrics.keys())
        values = [full_modality_metrics[metric_key]]
        
        for missing_mod, metrics in missing_modality_metrics.items():
            values.append(metrics[metric_key])
        
        x = np.arange(len(labels))
        ax.bar(x, values)
        ax.set_ylabel(metric_key)
        ax.set_title(f'{metric_key} Comparison: Full vs Missing Modalities')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        plt.tight_layout()
        self.writer.add_figure(f"{tag}/{metric_key}_comparison", fig, global_step)
        plt.close(fig)
    
    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Normalize images to [0, 1] range.
        
        Args:
            images: Tensor of images
        
        Returns:
            Normalized images
        """
        if images.dtype != torch.float32:
            images = images.float()
        
        min_val = images.min()
        max_val = images.max()
        
        if min_val == max_val:
            return torch.zeros_like(images)
        
        return (images - min_val) / (max_val - min_val)