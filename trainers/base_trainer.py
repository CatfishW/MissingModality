import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, List, Optional, Union, Any, Callable
from utils.distributed import is_main_process, reduce_tensor


class BaseTrainer:
    """
    Base trainer class with support for distributed training and tensorboard logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: torch.device = None,
        distributed: bool = False,
        output_dir: str = './output',
        tensorboard_dir: str = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            config: Configuration dictionary
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            criterion: Loss function
            optimizer: Optimizer (if None, will be created from config)
            scheduler: Learning rate scheduler
            device: Device to train on (if None, will use CUDA if available)
            distributed: Whether to use distributed training
            output_dir: Directory to save outputs
            tensorboard_dir: Directory for tensorboard logs (defaults to output_dir/tensorboard)
        """
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.output_dir = output_dir
        self.distributed = distributed
        
        # Set up device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Set up distributed training
        if self.distributed and dist.is_initialized():
            self.model = DDP(self.model, device_ids=[self.device], find_unused_parameters=True)
        
        # Set up optimizer if not provided
        if optimizer is None:
            self.setup_optimizer()
        else:
            self.optimizer = optimizer
        
        # Set up scheduler
        self.scheduler = scheduler
        
        # Set up directories
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up tensorboard
        if tensorboard_dir is None:
            tensorboard_dir = os.path.join(output_dir, 'tensorboard')
        
        if is_main_process():
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_dir)
        else:
            self.writer = None
        
        # Initialize best metrics for model saving
        self.best_val_metric = float('inf')
        self.best_epoch = 0
        
        # Visualization settings
        vis_config = self.config.get('visualization', {})
        self.vis_enabled = vis_config.get('enabled', True)
        self.vis_interval = vis_config.get('interval', 100)  # Every N batches
        self.vis_samples = vis_config.get('num_samples', 8)
        self.class_names = vis_config.get('class_names', None)
        
        # Import visualizer if visualization is enabled
        if self.vis_enabled and is_main_process() and self.writer is not None:
            from visualization.tensorboard import TensorboardVisualizer
            self.visualizer = TensorboardVisualizer(self.writer)
        else:
            self.visualizer = None
    
    def setup_optimizer(self):
        """
        Set up optimizer based on config.
        """
        optimizer_config = self.config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'adam').lower()
        lr = optimizer_config.get('lr', 1e-3)
        weight_decay = optimizer_config.get('weight_decay', 0)
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=lr, 
                momentum=momentum, 
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dict of metrics for this epoch
        """
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        start_time = time.time()
        
        try:
            for batch_idx, batch in enumerate(self.train_loader):
                # Move data to device
                batch = self._prepare_batch(batch)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                
                # Compute loss
                loss_dict = self._compute_loss(outputs, batch)
                loss = loss_dict['loss']
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                if self.distributed:
                    loss = reduce_tensor(loss)
                
                total_loss += loss.item() * batch['target'].size(0)
                total_samples += batch['target'].size(0)
                
                # Log batch stats for large datasets
                if batch_idx % self.config.get('log_interval', 10) == 0:
                    self._log_batch_stats(epoch, batch_idx, loss_dict)
                
                # Visualize current batch samples
                if self.vis_enabled and is_main_process() and self.visualizer is not None:
                    if batch_idx % self.vis_interval == 0:
                        self._visualize_batch(batch, outputs, f"train_epoch_{epoch}_batch_{batch_idx}", max_samples=self.vis_samples)
        except RuntimeError as e:
            if "DataLoader worker" in str(e):
                print(f"WARNING: DataLoader worker error occurred: {e}")
                # Recreate the DataLoader if possible
                if hasattr(self, '_recreate_data_loader'):
                    print("Attempting to recreate DataLoader...")
                    self._recreate_data_loader('train_loader')
                # If we haven't processed any batches, this is a critical error
                if total_samples == 0:
                    raise
            else:
                raise
        
        # Calculate epoch metrics
        if total_samples == 0:
            print("WARNING: No samples processed in this epoch!")
            return {'train_loss': float('inf'), 'train_time': time.time() - start_time}
            
        metrics = {
            'train_loss': total_loss / total_samples,
            'train_time': time.time() - start_time
        }
        
        # Update learning rate
        if self.scheduler is not None:
            if isinstance(
                self.scheduler, 
                (optim.lr_scheduler.ReduceLROnPlateau)
            ):
                self.scheduler.step(metrics['train_loss'])
            else:
                self.scheduler.step()
        
        return metrics
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Dict of validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Move data to device
                batch = self._prepare_batch(batch)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Compute loss
                loss_dict = self._compute_loss(outputs, batch)
                loss = loss_dict['loss']
                
                if self.distributed:
                    loss = reduce_tensor(loss)
                
                # Update metrics
                total_loss += loss.item() * batch['target'].size(0)
                total_samples += batch['target'].size(0)
                
                # Store predictions and targets for additional metrics
                preds = self._get_predictions(outputs)
                targets = batch['target']
                
                all_preds.append(preds.detach().cpu())
                all_targets.append(targets.detach().cpu())
                
                # Visualize validation samples
                if self.vis_enabled and is_main_process() and self.visualizer is not None:
                    if batch_idx % self.vis_interval == 0:
                        self._visualize_batch(batch, outputs, f"val_epoch_{epoch}_batch_{batch_idx}", max_samples=self.vis_samples)
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = {
            'val_loss': total_loss / total_samples
        }
        
        # Add task-specific metrics
        task_metrics = self._compute_task_metrics(all_preds, all_targets)
        metrics.update(task_metrics)
        
        return metrics
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Train the model for a specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train for
        
        Returns:
            Dict of metrics for all epochs
        """
        # Initialize metrics history
        metrics_history = {
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(1, num_epochs + 1):
            # Set epoch for distributed samplers
            if self.distributed:
                # Check if sampler has set_epoch method before calling it
                if hasattr(self.train_loader.sampler, 'set_epoch'):
                    self.train_loader.sampler.set_epoch(epoch)

            
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            metrics_history['train_loss'].append(train_metrics['train_loss'])
            
            # Validate
            val_metrics = self.validate(epoch)
            for k, v in val_metrics.items():
                if k not in metrics_history:
                    metrics_history[k] = []
                metrics_history[k].append(v)
            
            # Log epoch stats
            self._log_epoch_stats(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            if is_main_process():
                self.save_checkpoint(epoch, val_metrics)
        
        # Close tensorboard writer
        if self.writer is not None:
            self.writer.close()
        
        return metrics_history
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a batch for training/validation.
        
        Args:
            batch: Batch of data
        
        Returns:
            Processed batch with tensors moved to the correct device
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        return batch
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Compute loss for a batch.
        
        Args:
            outputs: Model outputs
            batch: Batch of data
        
        Returns:
            Dict of loss values
        """
        # Default implementation for classification
        loss = self.criterion(outputs['logits'], batch['target'])
        return {'loss': loss}
    
    def _get_predictions(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract predictions from model outputs.
        
        Args:
            outputs: Model outputs
        
        Returns:
            Tensor of predictions
        """
        # Default implementation for classification
        if 'preds' in outputs:
            return outputs['preds']
        elif 'logits' in outputs:
            return torch.argmax(outputs['logits'], dim=1)
        else:
            return outputs['output']
    
    def _compute_task_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute task-specific metrics.
        
        Args:
            preds: Model predictions
            targets: Ground truth
        
        Returns:
            Dict of metric names and values
        """
        # Default implementation for classification
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total
        
        return {'val_accuracy': accuracy}
    
    def _log_batch_stats(self, epoch: int, batch_idx: int, loss_dict: Dict[str, torch.Tensor]):
        """
        Log batch statistics during training.
        
        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            loss_dict: Dictionary of loss values
        """
        if not is_main_process():
            return
        
        # Calculate progress for logging
        dataset_size = len(self.train_loader.dataset)
        if self.distributed:
            world_size = dist.get_world_size()
            dataset_size = dataset_size // world_size
        
        batch_size = self.train_loader.batch_size
        progress = 100.0 * batch_idx * batch_size / dataset_size
        
        log_msg = f"Epoch: {epoch} [{batch_idx * batch_size}/{dataset_size} ({progress:.0f}%)] Loss: {loss_dict['loss'].item():.6f}"
        
        print(log_msg)
    
    def _log_epoch_stats(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """
        Log epoch statistics and write to tensorboard.
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        if not is_main_process():
            return
        
        # Print epoch summary
        print(f"Epoch {epoch} completed in {train_metrics['train_time']:.2f}s")
        print(f"Train Loss: {train_metrics['train_loss']:.6f}")
        
        for k, v in val_metrics.items():
            print(f"{k}: {v:.6f}")
        
        print("-" * 80)
        
        # Write to tensorboard
        if self.writer is not None:
            self.writer.add_scalar('Loss/train', train_metrics['train_loss'], epoch)
            
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"Metrics/{k}", v, epoch)
            
            # Add learning rate
            for i, param_group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(f'LR/param_group_{i}', param_group['lr'], epoch)
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
        """
        # Determine if this is the best model
        val_metric = metrics.get('val_loss', float('inf'))
        is_best = val_metric < self.best_val_metric
        
        if is_best:
            self.best_val_metric = val_metric
            self.best_epoch = epoch
            
            # Save best model
            model_to_save = self.model.module if self.distributed else self.model
            best_model_path = os.path.join(self.output_dir, 'best_model.pth')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
                'config': self.config
            }, best_model_path)
            
            print(f"Saved best model at epoch {epoch} with val_metric {val_metric:.6f}")
        
        # Save regular checkpoint
        if epoch % self.config.get('save_interval', 10) == 0:
            model_to_save = self.model.module if self.distributed else self.model
            checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pth')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
                'config': self.config
            }, checkpoint_path)
            
            print(f"Saved checkpoint at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str, strict: bool = True, 
                    reset_optimizer: bool = False, reset_lr_scheduler: bool = False,
                    reset_epoch: bool = False):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce that the keys match between state_dict and model
            reset_optimizer: Whether to reset the optimizer state
            reset_lr_scheduler: Whether to reset the learning rate scheduler
            reset_epoch: Whether to reset the epoch counter (start from 0)
        
        Returns:
            Checkpoint data
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model_to_load = self.model.module if self.distributed else self.model
        
        try:
            model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            print("Model weights loaded successfully")
        except Exception as e:
            print(f"Warning: Error loading model weights: {e}")
            if strict:
                raise
            
        # Load optimizer state unless reset is requested
        if 'optimizer_state_dict' in checkpoint and not reset_optimizer:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        else:
            print("Optimizer state reset (using current state)")
            
        # Load scheduler state if it exists and reset is not requested
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            if 'scheduler_state_dict' in checkpoint and not reset_lr_scheduler:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("Learning rate scheduler state loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load scheduler state: {e}")
            else:
                print("Learning rate scheduler state reset (using current state)")
        
        # Load epoch and metrics if they exist and reset is not requested
        if 'epoch' in checkpoint and not reset_epoch:
            self.best_epoch = checkpoint.get('best_epoch', checkpoint['epoch'])
            print(f"Resuming from epoch {checkpoint['epoch']} (best epoch: {self.best_epoch})")
        else:
            print("Epoch counter reset to 0")
            
        if 'metrics' in checkpoint and not reset_epoch:
            self.best_val_metric = checkpoint.get('best_val_metric', float('inf'))
            print(f"Best validation metric loaded: {self.best_val_metric:.6f}")
        
        print(f"Checkpoint loaded successfully from {checkpoint_path}")
        return checkpoint
    
    def _recreate_data_loader(self, loader_name: str):
        """
        Recreate a DataLoader that had worker failures.
        
        Args:
            loader_name: Name of the loader attribute to recreate ('train_loader' or 'val_loader')
        """
        if not hasattr(self, loader_name):
            print(f"No {loader_name} attribute found to recreate")
            return
            
        old_loader = getattr(self, loader_name)
        
        # Get the dataset and key parameters from the old loader
        dataset = old_loader.dataset
        batch_size = old_loader.batch_size
        
        # Use fewer workers and disable persistent workers to improve stability
        num_workers = max(0, getattr(old_loader, 'num_workers', 2) - 1)
        
        print(f"Recreating {loader_name} with {num_workers} workers (reduced) and persistent_workers=False")
        
        # Recreate the DataLoader with more conservative settings
        new_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=not isinstance(old_loader.sampler, torch.utils.data.DistributedSampler),
            sampler=old_loader.sampler,
            collate_fn=old_loader.collate_fn,
            pin_memory=getattr(old_loader, 'pin_memory', False),
            drop_last=getattr(old_loader, 'drop_last', False),
            persistent_workers=False  # Disable persistent workers
        )
        
        # Update the loader
        setattr(self, loader_name, new_loader)
    
    def _visualize_batch(self, batch: Dict[str, Any], outputs: Dict[str, torch.Tensor], tag: str, max_samples: int = 1):
        """
        Visualize a batch of data with predictions.
        
        Args:
            batch: Batch of data
            outputs: Model outputs
            tag: Tag for the visualization
            max_samples: Maximum number of samples to visualize (default: 1)
        """
        if not self.vis_enabled or self.visualizer is None:
            return
            
        # Get predictions
        preds = self._get_predictions(outputs)
        
        # Extract images and targets
        images = batch.get('image', None)
        targets = batch.get('target', None)
        
        # If we don't have images, we can't visualize
        if images is None:
            return
            
        # Hard-code max_samples to 1 to ensure we only show one image per batch
        max_samples = 1
            
        # Handle multimodal data - extract additional metadata if available
        metadata_list = None
        if 'text' in batch:
            # For hateful memes or similar multimodal datasets with text
            texts = batch.get('text', [])
            metadata_list = []
            for i, text in enumerate(texts):
                if i >= max_samples:
                    break
                metadata_list.append({'text': text})
        
        # Get class names if not provided
        class_names = self.class_names
        if class_names is None:
            # Try to infer class names from config
            model_cfg = self.config.get('model', {})
            if 'num_classes' in model_cfg:
                num_classes = model_cfg.get('num_classes')
                # Generate generic class names
                class_names = [f"Class {i}" for i in range(num_classes)]
                
                # Special case for binary classification like hateful memes
                if num_classes == 2:
                    class_names = ["Not Hateful", "Hateful"]
        
        # Visualize the batch (only one sample)
        self.visualizer.add_batch_with_predictions(
            tag=tag,
            images=images[:max_samples],
            ground_truths=targets[:max_samples] if targets is not None else None,
            predictions=preds[:max_samples],
            class_names=class_names,
            global_step=0,  # No global step needed for these visualizations
            max_samples=max_samples,
            metadata=metadata_list
        )