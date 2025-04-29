import torch
import numpy as np
# Set matplotlib backend to a non-interactive one before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use Agg backend which doesn't require a GUI
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
    
    def add_sample_with_prediction(
        self,
        tag: str,
        image: torch.Tensor,
        ground_truth: Any,
        prediction: Any,
        class_names: List[str] = None,
        global_step: int = 0,
        metadata: Dict[str, Any] = None
    ):
        """
        Add a single sample with its ground truth and prediction visualization to TensorBoard.
        
        Args:
            tag: Tag for the visualization
            image: Image tensor of shape (C, H, W)
            ground_truth: Ground truth label or class index
            prediction: Predicted label or class index
            class_names: Optional list of class names for label mapping
            global_step: Global step value
            metadata: Additional metadata to display (e.g., text associated with image)
        """
        # Create a figure and add the image
        fig = plt.figure(figsize=(12, 8))
        
        # Add image to the plot
        ax_img = fig.add_subplot(1, 1, 1)
        
        # If image is a tensor, convert to numpy and transpose for visualization
        if isinstance(image, torch.Tensor):
            # If the image has 3 dimensions, assume it's (C, H, W) format
            if image.dim() == 3:
                # Convert to numpy and transpose to (H, W, C)
                img_np = image.detach().cpu().numpy()
                if img_np.shape[0] in [1, 3]:  # If channels dimension is first
                    img_np = np.transpose(img_np, (1, 2, 0))
                
                # If single channel, squeeze or repeat to make it displayable
                if img_np.shape[-1] == 1:
                    img_np = np.squeeze(img_np)
            else:
                img_np = image.detach().cpu().numpy()
        else:
            img_np = image
        
        # Normalize image for better visualization
        if img_np.min() < 0 or img_np.max() > 1:
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        # Show image
        ax_img.imshow(img_np)
        ax_img.axis('off')  # Turn off axis
        
        # Map class indices to names if provided
        gt_label = ground_truth
        pred_label = prediction
        
        if class_names is not None:
            if isinstance(ground_truth, (int, torch.Tensor, np.ndarray)):
                if isinstance(ground_truth, torch.Tensor):
                    gt_idx = ground_truth.item() if ground_truth.numel() == 1 else ground_truth.argmax().item()
                elif isinstance(ground_truth, np.ndarray):
                    gt_idx = ground_truth.item() if ground_truth.size == 1 else ground_truth.argmax().item()
                else:
                    gt_idx = ground_truth
                gt_label = class_names[gt_idx]
            
            if isinstance(prediction, (int, torch.Tensor, np.ndarray)):
                if isinstance(prediction, torch.Tensor):
                    pred_idx = prediction.item() if prediction.numel() == 1 else prediction.argmax().item()
                elif isinstance(prediction, np.ndarray):
                    pred_idx = prediction.item() if prediction.size == 1 else prediction.argmax().item()
                else:
                    pred_idx = prediction
                pred_label = class_names[pred_idx]
        
        # Create title with ground truth and prediction info
        title = f"GT: {gt_label} | Pred: {pred_label}"
        
        # Set a title color based on correctness
        title_color = 'green' if gt_label == pred_label else 'red'
        
        # Add metadata text if provided
        if metadata:
            metadata_text = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
            title = f"{title}\n{metadata_text}"
        
        ax_img.set_title(title, color=title_color)
        
        # Add the figure to tensorboard
        self.writer.add_figure(tag, fig, global_step)
        plt.close(fig)
        
    def add_batch_with_predictions(
        self,
        tag: str,
        images: torch.Tensor,
        ground_truths: torch.Tensor,
        predictions: torch.Tensor,
        class_names: List[str] = None,
        global_step: int = 0,
        max_samples: int = 8,
        metadata: List[Dict[str, Any]] = None
    ):
        """
        Add a batch of samples with their ground truths and predictions to TensorBoard.
        
        Args:
            tag: Base tag for the visualizations
            images: Batch of images (B, C, H, W)
            ground_truths: Batch of ground truth labels
            predictions: Batch of predicted labels
            class_names: Optional list of class names for label mapping
            global_step: Global step value
            max_samples: Maximum number of samples to visualize
            metadata: Optional list of metadata dictionaries for each sample
        """
        # Ensure we don't exceed the batch size
        batch_size = min(len(images), max_samples)
        
        for i in range(batch_size):
            sample_tag = f"{tag}/sample_{i}"
            sample_metadata = metadata[i] if metadata and i < len(metadata) else None
            
            self.add_sample_with_prediction(
                sample_tag,
                images[i],
                ground_truths[i] if len(ground_truths) > i else None,
                predictions[i] if len(predictions) > i else None,
                class_names=class_names,
                global_step=global_step,
                metadata=sample_metadata
            )
    
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