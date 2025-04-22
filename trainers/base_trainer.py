import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
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
        
        # Calculate epoch metrics
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
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model_to_load = self.model.module if self.distributed else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint