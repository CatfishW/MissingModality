import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

from trainers.base_trainer import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    """
    Trainer for classification tasks with missing modality support.
    """
    
    def __init__(self, num_classes: int, **kwargs):
        """
        Initialize the classification trainer.
        
        Args:
            num_classes: Number of classes for classification
            **kwargs: Arguments to pass to BaseTrainer
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for classification.
        
        Args:
            outputs: Model outputs
            batch: Batch of data
        
        Returns:
            Dict of loss values
        """
        loss = self.criterion(outputs['logits'], batch['target'])
        return {'loss': loss}
    
    def _compute_task_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute classification-specific metrics.
        
        Args:
            preds: Model predictions
            targets: Ground truth
        
        Returns:
            Dict of metric names and values
        """
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total
        
        # Compute per-class accuracy if number of classes is reasonable
        if self.num_classes <= 20:
            per_class_correct = torch.zeros(self.num_classes, device=preds.device)
            per_class_total = torch.zeros(self.num_classes, device=preds.device)
            
            for c in range(self.num_classes):
                per_class_correct[c] = ((preds == c) & (targets == c)).sum().item()
                per_class_total[c] = (targets == c).sum().item()
            
            # Avoid division by zero
            per_class_acc = per_class_correct / per_class_total.clamp(min=1)
            class_avg_acc = per_class_acc.mean().item()
            
            return {
                'val_accuracy': accuracy,
                'val_class_avg_accuracy': class_avg_acc
            }
        
        return {'val_accuracy': accuracy}


class RegressionTrainer(BaseTrainer):
    """
    Trainer for regression tasks with missing modality support.
    """
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for regression.
        
        Args:
            outputs: Model outputs
            batch: Batch of data
        
        Returns:
            Dict of loss values
        """
        loss = self.criterion(outputs['output'], batch['target'])
        return {'loss': loss}
    
    def _get_predictions(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract predictions from model outputs for regression.
        
        Args:
            outputs: Model outputs
        
        Returns:
            Tensor of predictions
        """
        if 'preds' in outputs:
            return outputs['preds']
        else:
            return outputs['output']
    
    def _compute_task_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute regression-specific metrics.
        
        Args:
            preds: Model predictions
            targets: Ground truth
        
        Returns:
            Dict of metric names and values
        """
        # Mean Squared Error
        mse = ((preds - targets) ** 2).mean().item()
        
        # Mean Absolute Error
        mae = torch.abs(preds - targets).mean().item()
        
        # Root Mean Squared Error
        rmse = torch.sqrt(torch.tensor(mse)).item()
        
        return {
            'val_mse': mse,
            'val_mae': mae,
            'val_rmse': rmse
        }


class DetectionTrainer(BaseTrainer):
    """
    Trainer for object detection tasks with missing modality support.
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.5,
        score_threshold: float = 0.05,
        **kwargs
    ):
        """
        Initialize the detection trainer.
        
        Args:
            iou_threshold: IoU threshold for detection evaluation
            score_threshold: Score threshold for detection evaluation
            **kwargs: Arguments to pass to BaseTrainer
        """
        super().__init__(**kwargs)
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for object detection.
        
        Args:
            outputs: Model outputs containing losses
            batch: Batch of data
        
        Returns:
            Dict of loss values
        """
        # For object detection, the model typically computes the losses internally
        # due to the complexity of detection losses
        loss_dict = outputs.get('losses', {})
        
        # If no losses are provided, compute them using the criterion
        if not loss_dict:
            loss = self.criterion(outputs, batch)
            loss_dict = {'loss': loss}
        
        # Compute total loss
        if 'loss' not in loss_dict:
            loss_dict['loss'] = sum(loss for loss in loss_dict.values())
        
        return loss_dict
    
    def _get_predictions(self, outputs: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Extract predictions from model outputs for detection.
        
        Args:
            outputs: Model outputs
        
        Returns:
            List of predictions for each image in the batch
        """
        if 'detections' in outputs:
            return outputs['detections']
        else:
            return outputs['preds']
    
    def _compute_task_metrics(
        self, 
        preds: List[Dict[str, torch.Tensor]], 
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Compute detection-specific metrics (mAP, precision, recall).
        
        Args:
            preds: List of predictions for each image
            targets: List of targets for each image
        
        Returns:
            Dict of metric names and values
        """
        # This is a simplified calculation of mAP
        # In practice, you would use a more robust implementation
        # like the one in COCO API or torchvision's detection evaluation
        
        # Initialize metrics
        total_precision = 0.0
        total_recall = 0.0
        total_images = len(preds)
        
        for pred, target in zip(preds, targets):
            # Calculate precision and recall for this image
            precision, recall = self._calculate_precision_recall(pred, target)
            
            total_precision += precision
            total_recall += recall
        
        # Calculate average precision and recall
        avg_precision = total_precision / total_images
        avg_recall = total_recall / total_images
        
        # Calculate F1 score
        if avg_precision + avg_recall > 0:
            f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        else:
            f1_score = 0.0
        
        return {
            'val_precision': avg_precision,
            'val_recall': avg_recall,
            'val_f1': f1_score
        }
    
    def _calculate_precision_recall(
        self, 
        pred: Dict[str, torch.Tensor], 
        target: Dict[str, torch.Tensor]
    ) -> Tuple[float, float]:
        """
        Calculate precision and recall for a single image.
        
        Args:
            pred: Predictions for an image (boxes, scores, labels)
            target: Targets for an image (boxes, labels)
        
        Returns:
            Tuple of (precision, recall)
        """
        # Get predictions above score threshold
        if 'scores' in pred:
            keep = pred['scores'] > self.score_threshold
            pred_boxes = pred['boxes'][keep]
            pred_labels = pred['labels'][keep]
        else:
            pred_boxes = pred['boxes']
            pred_labels = pred['labels']
        
        target_boxes = target['boxes']
        target_labels = target['labels']
        
        # No predictions or no targets
        if len(pred_boxes) == 0 and len(target_boxes) == 0:
            return 1.0, 1.0  # Perfect score
        elif len(pred_boxes) == 0:
            return 0.0, 0.0  # No predictions
        elif len(target_boxes) == 0:
            return 0.0, 1.0  # No targets but we made predictions
        
        # Calculate IoU between each pred and target box
        ious = self._box_iou(pred_boxes, target_boxes)
        
        # For each target, find the pred with highest IoU above threshold
        matched_preds = set()
        matched_targets = set()
        
        for t_idx in range(len(target_boxes)):
            # Consider only predictions with matching class
            matching_class_mask = (pred_labels == target_labels[t_idx])
            if not matching_class_mask.any():
                continue
                
            # Find best matching prediction for this target
            valid_ious = ious[:, t_idx].clone()
            valid_ious[~matching_class_mask] = -1  # Mask out non-matching classes
            
            if valid_ious.max() > self.iou_threshold:
                p_idx = valid_ious.argmax().item()
                matched_preds.add(p_idx)
                matched_targets.add(t_idx)
        
        # Calculate precision and recall
        true_positives = len(matched_preds)
        false_positives = len(pred_boxes) - true_positives
        false_negatives = len(target_boxes) - len(matched_targets)
        
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        
        return precision, recall
    
    def _box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between two sets of boxes.
        
        Args:
            boxes1: First set of boxes (N, 4) in [x1, y1, x2, y2] format
            boxes2: Second set of boxes (M, 4) in [x1, y1, x2, y2] format
        
        Returns:
            IoU matrix of shape (N, M)
        """
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Calculate intersection areas
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # (N, M, 2)
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # (N, M, 2)
        
        wh = (rb - lt).clamp(min=0)  # (N, M, 2)
        intersection = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
        
        union = area1[:, None] + area2 - intersection
        
        iou = intersection / union
        
        return iou