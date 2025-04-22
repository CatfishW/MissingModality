import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Union, Optional

from utils.config import Config
from utils.distributed import setup_distributed, set_seed
from data.datasets.base import MissingModalityDataset


def parse_args():
    parser = argparse.ArgumentParser(description='MissingModality Testing')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Path to save results')
    parser.add_argument('--missing', type=str, nargs='+', default=None, 
                        help='List of modalities to exclude during testing')
    parser.add_argument('--visualize', action='store_true', help='Visualize results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
    
    Returns:
        Model instance and configuration
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint or create a default one
    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            config = Config(config_dict=config)
    else:
        config = Config()
    
    # Build model
    model = build_model(config)
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, config


def build_model(config):
    """
    Build model based on configuration.
    
    Args:
        config: Configuration object
    
    Returns:
        Model instance
    """
    model_cfg = config.get('model', {})
    model_type = model_cfg.get('type', 'classification')
    
    if model_type == 'classification':
        from models.classification_model import MultiModalClassifier
        model = MultiModalClassifier(
            modalities=config['data']['modalities'],
            num_classes=model_cfg.get('num_classes', 2),
            modality_dimensions=model_cfg.get('modality_dimensions', {}),
            fusion_type=model_cfg.get('fusion_type', 'attention'),
            fusion_dimension=model_cfg.get('fusion_dimension', 512),
            dropout=model_cfg.get('dropout', 0.2),
            pretrained=False,  # Don't need pretrained for testing
            handling_missing=model_cfg.get('handling_missing', {'strategy': 'zero'})
        )
    elif model_type == 'detection':
        from models.detection_model import MultiModalDetector
        model = MultiModalDetector(
            modalities=config['data']['modalities'],
            num_classes=model_cfg.get('num_classes', 80),
            modality_dimensions=model_cfg.get('modality_dimensions', {}),
            fusion_type=model_cfg.get('fusion_type', 'attention'),
            fusion_dimension=model_cfg.get('fusion_dimension', 1024),
            dropout=model_cfg.get('dropout', 0.1),
            pretrained=False,  # Don't need pretrained for testing
            handling_missing=model_cfg.get('handling_missing', {'strategy': 'zero'}),
            detection_cfg=model_cfg.get('detection', {})
        )
    elif model_type == 'regression':
        from models.regression_model import MultiModalRegressor
        model = MultiModalRegressor(
            modalities=config['data']['modalities'],
            output_dim=model_cfg.get('output_dim', 1),
            modality_dimensions=model_cfg.get('modality_dimensions', {}),
            fusion_type=model_cfg.get('fusion_type', 'attention'),
            fusion_dimension=model_cfg.get('fusion_dimension', 512),
            dropout=model_cfg.get('dropout', 0.2),
            pretrained=False,  # Don't need pretrained for testing
            handling_missing=model_cfg.get('handling_missing', {'strategy': 'zero'})
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model


def prepare_test_dataset(config, missing_modalities=None):
    """
    Prepare dataset for testing with potentially missing modalities.
    
    Args:
        config: Configuration object
        missing_modalities: List of modalities to exclude during testing
        
    Returns:
        Test dataset
    """
    data_cfg = config.get('data', {})
    modalities = data_cfg.get('modalities', [])
    
    # Import your dataset classes
    from data.datasets.your_dataset import YourMultiModalDataset
    
    # Create test dataset
    test_data_cfg = data_cfg.get('test_data', {})
    test_dataset = YourMultiModalDataset(
        data_path=test_data_cfg.get('path', './data'),
        modalities=modalities,
        split='test'
    )
    
    # Wrap with MissingModalityDataset to simulate missing modalities
    if missing_modalities:
        test_dataset = MissingModalityDataset(
            dataset=test_dataset,
            missing_modalities=missing_modalities
        )
    
    return test_dataset


def prepare_dataloader(config, dataset):
    """
    Prepare dataloader for testing.
    
    Args:
        config: Configuration object
        dataset: Test dataset
    
    Returns:
        Test dataloader
    """
    from data.loaders.dataloader import create_dataloader, default_collate_fn
    
    data_cfg = config.get('data', {})
    test_cfg = data_cfg.get('test_data', {})
    
    return create_dataloader(
        dataset=dataset,
        batch_size=test_cfg.get('batch_size', 32),
        num_workers=test_cfg.get('num_workers', 4),
        shuffle=False,
        drop_last=False,
        distributed=False,
        collate_fn=default_collate_fn,
        pin_memory=True
    )


def evaluate_classification(model, dataloader, device, num_classes):
    """
    Evaluate classification model.
    
    Args:
        model: Model to evaluate
        dataloader: Test dataloader
        device: Device
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # Move data to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Extract predictions
            if 'logits' in outputs:
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=1)
                all_logits.append(logits.cpu())
            else:
                preds = outputs['preds']
            
            all_preds.append(preds.cpu())
            all_targets.append(batch['target'].cpu())
    
    # Concatenate predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    accuracy = (all_preds == all_targets).float().mean().item()
    
    # Calculate per-class accuracy
    per_class_acc = []
    for i in range(num_classes):
        idx = (all_targets == i)
        if idx.sum() > 0:
            class_acc = (all_preds[idx] == all_targets[idx]).float().mean().item()
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)
    
    # Calculate confusion matrix
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(all_targets, all_preds):
        conf_matrix[t.item(), p.item()] += 1
    
    # Calculate confusion matrix based metrics
    tp = torch.diag(conf_matrix)
    fp = conf_matrix.sum(dim=0) - tp
    fn = conf_matrix.sum(dim=1) - tp
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    # Compute logits-based metrics if available
    if all_logits:
        all_logits = torch.cat(all_logits, dim=0)
        # Convert to probabilities
        probs = torch.softmax(all_logits, dim=1)
        # Calculate cross-entropy loss
        loss = torch.nn.functional.cross_entropy(all_logits, all_targets).item()
    else:
        loss = 0.0
    
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'class_avg_accuracy': sum(per_class_acc) / len(per_class_acc),
        'precision': precision.mean().item(),
        'recall': recall.mean().item(),
        'f1': f1.mean().item(),
        'loss': loss,
        'confusion_matrix': conf_matrix.tolist()
    }


def evaluate_regression(model, dataloader, device):
    """
    Evaluate regression model.
    
    Args:
        model: Model to evaluate
        dataloader: Test dataloader
        device: Device
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # Move data to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Extract predictions
            if 'output' in outputs:
                preds = outputs['output']
            else:
                preds = outputs['preds']
            
            all_preds.append(preds.cpu())
            all_targets.append(batch['target'].cpu())
    
    # Concatenate predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    mse = torch.nn.functional.mse_loss(all_preds, all_targets).item()
    mae = torch.nn.functional.l1_loss(all_preds, all_targets).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    
    # Calculate R^2 (coefficient of determination)
    target_mean = all_targets.mean()
    ss_tot = ((all_targets - target_mean) ** 2).sum()
    ss_res = ((all_targets - all_preds) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def evaluate_detection(model, dataloader, device, iou_threshold=0.5, score_threshold=0.05):
    """
    Evaluate detection model.
    
    Args:
        model: Model to evaluate
        dataloader: Test dataloader
        device: Device
        iou_threshold: IoU threshold for detection evaluation
        score_threshold: Score threshold for detection evaluation
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    all_precisions = []
    all_recalls = []
    all_f1s = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # Move data to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Forward pass
            outputs = model(batch)
            
            # Extract predictions
            if 'detections' in outputs:
                detections = outputs['detections']
            else:
                detections = outputs['preds']
            
            # Calculate metrics for this batch
            for pred, target in zip(detections, batch['target']):
                # Calculate precision and recall for this image
                precision, recall = calculate_precision_recall(
                    pred, target, iou_threshold, score_threshold
                )
                
                all_precisions.append(precision)
                all_recalls.append(recall)
                
                # Calculate F1
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0
                
                all_f1s.append(f1)
    
    # Calculate average metrics
    avg_precision = sum(all_precisions) / max(len(all_precisions), 1)
    avg_recall = sum(all_recalls) / max(len(all_recalls), 1)
    avg_f1 = sum(all_f1s) / max(len(all_f1s), 1)
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1
    }


def calculate_precision_recall(pred, target, iou_threshold=0.5, score_threshold=0.05):
    """
    Calculate precision and recall for detection task.
    
    Args:
        pred: Predictions for an image
        target: Targets for an image
        iou_threshold: IoU threshold
        score_threshold: Score threshold
    
    Returns:
        Tuple of (precision, recall)
    """
    # Filter predictions by score threshold
    if 'scores' in pred:
        keep = pred['scores'] > score_threshold
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
    ious = box_iou(pred_boxes, target_boxes)
    
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
        
        if valid_ious.max() > iou_threshold:
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


def box_iou(boxes1, boxes2):
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


def visualize_missing_modality_results(results, output_dir):
    """
    Visualize the results of testing with missing modalities.
    
    Args:
        results: Dictionary mapping scenario to metrics
        output_dir: Directory to save visualizations
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the keys based on the first result's metrics
    first_scenario = list(results.keys())[0]
    metric_keys = [k for k in results[first_scenario].keys() 
                  if not k.startswith('per_') and not k.startswith('confusion_')]
    
    # Prepare the data for plotting
    data = []
    for scenario, metrics in results.items():
        for metric in metric_keys:
            if metric in metrics:
                data.append({
                    'Scenario': scenario,
                    'Metric': metric,
                    'Value': metrics[metric]
                })
    
    # Create a DataFrame
    df = pd.DataFrame(data)
    
    # Plot for each metric
    for metric in metric_keys:
        plt.figure(figsize=(10, 6))
        metric_data = df[df['Metric'] == metric]
        
        sns.barplot(data=metric_data, x='Scenario', y='Value')
        plt.title(f'Effect of Missing Modalities on {metric}')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'))
        plt.close()
    
    # Also save as CSV for further analysis
    df.to_csv(os.path.join(output_dir, 'missing_modality_results.csv'), index=False)
    
    print(f"Visualizations saved to {output_dir}")


def main():
    """
    Main function to test with missing modalities.
    """
    # Parse arguments
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Load config
    config = Config(args.config)
    
    # Set random seed for reproducibility
    set_seed(config.get('seed', 42))
    
    # Load model from checkpoint
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    
    # Get modalities
    modalities = config['data']['modalities']
    
    # Get model type
    model_type = config['model']['type']
    
    # Scenarios to test
    scenarios = {
        'full': None  # Test with all modalities
    }
    
    # Add user-specified missing modalities
    if args.missing:
        for modality in args.missing:
            if modality in modalities:
                scenarios[f'missing_{modality}'] = [modality]
    else:
        # Otherwise test with each modality missing
        for modality in modalities:
            scenarios[f'missing_{modality}'] = [modality]
        
        # Add a scenario with all modalities missing except one
        if len(modalities) > 2:
            for modality in modalities:
                missing_all_except_one = [m for m in modalities if m != modality]
                scenarios[f'only_{modality}'] = missing_all_except_one
    
    # Prepare output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(os.path.dirname(args.checkpoint), 'missing_modality_results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store results for each scenario
    all_results = {}
    
    # Test each scenario
    for scenario_name, missing_mods in scenarios.items():
        print(f"\nEvaluating scenario: {scenario_name}")
        if missing_mods:
            print(f"Missing modalities: {missing_mods}")
        else:
            print("All modalities present")
        
        # Prepare dataset and dataloader
        test_dataset = prepare_test_dataset(config, missing_mods)
        test_loader = prepare_dataloader(config, test_dataset)
        
        # Evaluate based on model type
        if model_type == 'classification':
            num_classes = config['model'].get('num_classes', 2)
            results = evaluate_classification(model, test_loader, device, num_classes)
        elif model_type == 'regression':
            results = evaluate_regression(model, test_loader, device)
        elif model_type == 'detection':
            iou_threshold = config.get('training', {}).get('eval_settings', {}).get('iou_threshold', 0.5)
            score_threshold = config.get('training', {}).get('eval_settings', {}).get('score_threshold', 0.05)
            results = evaluate_detection(model, test_loader, device, iou_threshold, score_threshold)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Store results
        all_results[scenario_name] = results
        
        # Print results
        print(f"Results for {scenario_name}:")
        for k, v in results.items():
            if not k.startswith('per_') and not k.startswith('confusion_'):
                print(f"  {k}: {v:.4f}")
    
    # Save all results
    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Visualize results if requested
    if args.visualize:
        visualize_missing_modality_results(all_results, output_dir)
    
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()