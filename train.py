import os
import sys
import torch
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.config import Config
from utils.distributed import setup_distributed, set_seed
from trainers.task_trainers import ClassificationTrainer, RegressionTrainer, DetectionTrainer

#python train.py --config configs/hateful_memes.yaml
def parse_args():
    parser = argparse.ArgumentParser(description='MissingModality Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()


def build_model(config):
    """
    Build model based on configuration.
    
    Args:
        config: Configuration object
    
    Returns:
        Model instance
    """
    model_cfg = config.get('model', {})
    model_type = model_cfg.get('type', 'hateful_memes')
    print(f"Building model of type: {model_type}")
    
    if model_type == 'hateful_memes':
        from models.classification_model import HatefulMemesClassifier
        model = HatefulMemesClassifier(
            num_classes=model_cfg.get('num_classes', 2),
            image_encoder=model_cfg.get('image_encoder', 'resnet50'),
            text_encoder=model_cfg.get('text_encoder', 'bert-base-uncased'),
            fusion_type=model_cfg.get('fusion_type', 'attention'),
            fusion_dim=model_cfg.get('fusion_dim', 512),
            dropout=model_cfg.get('dropout', 0.2),
            pretrained=model_cfg.get('pretrained', True),
            handling_missing=model_cfg.get('handling_missing', {'strategy': 'zero'})
        )
    elif model_type == 'classification':
        from models.classification_model import MultiModalClassifier  # Import your model implementation
        model = MultiModalClassifier(
            modalities=config['data']['modalities'],
            num_classes=model_cfg.get('num_classes', 2),
            modality_dimensions=model_cfg.get('modality_dimensions', {}),
            fusion_type=model_cfg.get('fusion_type', 'attention'),
            fusion_dimension=model_cfg.get('fusion_dimension', 512),
            dropout=model_cfg.get('dropout', 0.2),
            pretrained=model_cfg.get('pretrained', True),
            handling_missing=model_cfg.get('handling_missing', {'strategy': 'zero'})
        )
    elif model_type == 'detection':
        from models.detection_model import MultiModalDetector  # Import your model implementation
        model = MultiModalDetector(
            modalities=config['data']['modalities'],
            num_classes=model_cfg.get('num_classes', 80),
            modality_dimensions=model_cfg.get('modality_dimensions', {}),
            fusion_type=model_cfg.get('fusion_type', 'attention'),
            fusion_dimension=model_cfg.get('fusion_dimension', 1024),
            dropout=model_cfg.get('dropout', 0.1),
            pretrained=model_cfg.get('pretrained', True),
            handling_missing=model_cfg.get('handling_missing', {'strategy': 'zero'}),
            detection_cfg=model_cfg.get('detection', {})
        )
    elif model_type == 'regression':
        from models.regression_model import MultiModalRegressor  # Import your model implementation
        model = MultiModalRegressor(
            modalities=config['data']['modalities'],
            output_dim=model_cfg.get('output_dim', 1),
            modality_dimensions=model_cfg.get('modality_dimensions', {}),
            fusion_type=model_cfg.get('fusion_type', 'attention'),
            fusion_dimension=model_cfg.get('fusion_dimension', 512),
            dropout=model_cfg.get('dropout', 0.2),
            pretrained=model_cfg.get('pretrained', True),
            handling_missing=model_cfg.get('handling_missing', {'strategy': 'zero'})
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model


def build_datasets(config):
    """
    Build datasets based on configuration.
    
    Args:
        config: Configuration object
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    data_cfg = config.get('data', {})
    modalities = data_cfg.get('modalities', [])
    model_type = config.get('model', {}).get('type', 'classification')
    dataset_source = data_cfg.get('dataset_source', 'local')  # 'local' or 'huggingface'
    
    # Import dataset and missing modality handling
    from data.datasets.base import MissingModalityDataset
    
    if model_type == 'hateful_memes':
        # Import hateful memes dataset and transforms
        from data.transforms.hateful_memes_transforms import HatefulMemesTransforms
        
        # Set up transformations
        image_transform_cfg = config.get('image_transforms', {})
        text_transform_cfg = config.get('text_transforms', {})
        
        # Image transforms
        train_image_transform = HatefulMemesTransforms.get_image_transform(
            split='train',
            image_size=image_transform_cfg.get('train', {}).get('image_size', 224),
            normalize=image_transform_cfg.get('train', {}).get('normalize', True)
        )
        
        val_image_transform = HatefulMemesTransforms.get_image_transform(
            split='val',
            image_size=image_transform_cfg.get('val', {}).get('image_size', 224),
            normalize=image_transform_cfg.get('val', {}).get('normalize', True)
        )
        
        test_image_transform = HatefulMemesTransforms.get_image_transform(
            split='test',
            image_size=image_transform_cfg.get('test', {}).get('image_size', 224),
            normalize=image_transform_cfg.get('test', {}).get('normalize', True)
        )
        
        # Text transforms
        text_transform = HatefulMemesTransforms.get_text_transform(
            model_name=text_transform_cfg.get('model_name', 'bert-base-uncased'),
            max_length=text_transform_cfg.get('max_length', 128)
        )
        
        # Create datasets based on source
        if dataset_source == 'huggingface':
            # Use Hugging Face dataset
            from data.datasets.huggingface_hateful_memes import HuggingFaceHatefulMemesDataset
            
            dataset_name = data_cfg.get('dataset_name', 'neuralcatcher/hateful_memes')
            cache_dir = data_cfg.get('cache_dir', None)
            data_dir = data_cfg.get('data_path', './data')
            
            print(f"Loading Hateful Memes dataset from Hugging Face: {dataset_name}")
            print(f"Using data directory for images: {data_dir}")
            
            train_dataset = HuggingFaceHatefulMemesDataset(
                split='train',
                transform=train_image_transform,
                text_transform=text_transform,
                dataset_name=dataset_name,
                cache_dir=cache_dir,
                data_dir=data_dir
            )
            
            val_dataset = HuggingFaceHatefulMemesDataset(
                split='val',
                transform=val_image_transform,
                text_transform=text_transform,
                dataset_name=dataset_name,
                cache_dir=cache_dir,
                data_dir=data_dir
            )
            
            test_dataset = HuggingFaceHatefulMemesDataset(
                split='test',
                transform=test_image_transform,
                text_transform=text_transform,
                dataset_name=dataset_name,
                cache_dir=cache_dir,
                data_dir=data_dir
            )
        else:
            # Use local dataset
            from data.datasets.hateful_memes import HatefulMemesDataset
            
            # Get data path
            data_path = data_cfg.get('data_path', './data')
            
            print(f"Loading Hateful Memes dataset from local path: {data_path}")
            
            train_dataset = HatefulMemesDataset(
                data_path=data_path,
                split='train',
                transform=train_image_transform,
                text_transform=text_transform
            )
            
            val_dataset = HatefulMemesDataset(
                data_path=data_path,
                split='val',
                transform=val_image_transform,
                text_transform=text_transform
            )
            
            test_dataset = HatefulMemesDataset(
                data_path=data_path,
                split='test',
                transform=test_image_transform,
                text_transform=text_transform
            )
    else:
        # Original dataset loading code
        # Import appropriate dataset classes
        try:
            from data.datasets.your_dataset import YourMultiModalDataset
        except ImportError:
            raise ImportError("You need to implement YourMultiModalDataset for your specific data")
        
        # Create train dataset
        train_data_cfg = data_cfg.get('train_data', {})
        train_dataset = YourMultiModalDataset(
            data_path=train_data_cfg.get('path', './data'),
            modalities=modalities,
            split='train'
        )
        
        # Create validation dataset
        val_data_cfg = data_cfg.get('val_data', {})
        val_dataset = YourMultiModalDataset(
            data_path=val_data_cfg.get('path', './data'),
            modalities=modalities,
            split='val'
        )
        
        # Create test dataset
        test_data_cfg = data_cfg.get('test_data', {})
        test_dataset = YourMultiModalDataset(
            data_path=test_data_cfg.get('path', './data'),
            modalities=modalities,
            split='test'
        )
    
    # Wrap with MissingModalityDataset if enabled
    missing_cfg = data_cfg.get('missing_modalities', {})
    if missing_cfg.get('enabled', False):
        # For training, use random missing with specified probability
        train_missing_prob = missing_cfg.get('train_missing_prob', 0.0)
        if train_missing_prob > 0:
            train_dataset = MissingModalityDataset(
                dataset=train_dataset,
                missing_prob=train_missing_prob,
                random_missing=True
            )
    
    return train_dataset, val_dataset, test_dataset


def build_dataloaders(config, train_dataset, val_dataset, test_dataset, distributed=False):
    """
    Build dataloaders based on configuration.
    
    Args:
        config: Configuration object
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        distributed: Whether to use distributed training
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from data.loaders.dataloader import create_dataloader, default_collate_fn
    
    data_cfg = config.get('data', {})
    train_cfg = data_cfg.get('train_data', {})
    val_cfg = data_cfg.get('val_data', {})
    test_cfg = data_cfg.get('test_data', {})
    
    # Create dataloaders
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=train_cfg.get('batch_size', 32),
        num_workers=train_cfg.get('num_workers', 4),
        shuffle=True,
        drop_last=True,
        distributed=distributed,
        collate_fn=default_collate_fn,
        pin_memory=True
    )
    
    val_loader = create_dataloader(
        dataset=val_dataset,
        batch_size=val_cfg.get('batch_size', 32),
        num_workers=val_cfg.get('num_workers', 4),
        shuffle=False,
        drop_last=False,
        distributed=distributed,
        collate_fn=default_collate_fn,
        pin_memory=True
    )
    
    test_loader = create_dataloader(
        dataset=test_dataset,
        batch_size=test_cfg.get('batch_size', 32),
        num_workers=test_cfg.get('num_workers', 4),
        shuffle=False,
        drop_last=False,
        distributed=distributed,
        collate_fn=default_collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def build_criterion(config):
    """
    Build loss criterion based on configuration.
    
    Args:
        config: Configuration object
    
    Returns:
        Loss criterion
    """
    import torch.nn as nn
    
    training_cfg = config.get('training', {})
    criterion_name = training_cfg.get('criterion', 'cross_entropy')
    model_type = config['model']['type']
    
    if model_type == 'classification':
        if criterion_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif criterion_name == 'bce':
            return nn.BCEWithLogitsLoss()
        elif criterion_name == 'focal':
            from utils.losses import FocalLoss
            return FocalLoss(alpha=0.25, gamma=2.0)
    elif model_type == 'regression':
        if criterion_name == 'mse':
            return nn.MSELoss()
        elif criterion_name == 'l1':
            return nn.L1Loss()
        elif criterion_name == 'smooth_l1':
            return nn.SmoothL1Loss()
    elif model_type == 'detection':
        # For detection, typically the model handles the loss calculation internally
        return lambda x, y: sum(loss for loss in x['losses'].values())
    elif model_type == "hateful_memes":
        if criterion_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif criterion_name == 'bce':
            return nn.BCEWithLogitsLoss()
        elif criterion_name == 'focal':
            from utils.losses import FocalLoss
            return FocalLoss(alpha=0.25, gamma=2.0)
    
    raise ValueError(f"Unsupported criterion: {criterion_name} for model type: {model_type}")


def build_optimizer(config, model):
    """
    Build optimizer based on configuration.
    
    Args:
        config: Configuration object
        model: Model instance
    
    Returns:
        Optimizer
    """
    import torch.optim as optim
    
    training_cfg = config.get('training', {})
    optimizer_cfg = training_cfg.get('optimizer', {})
    
    name = optimizer_cfg.get('name', 'adam')
    lr = optimizer_cfg.get('lr', 0.001)
    weight_decay = optimizer_cfg.get('weight_decay', 0.0001)
    
    if name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        momentum = optimizer_cfg.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(config, optimizer):
    """
    Build learning rate scheduler based on configuration.
    
    Args:
        config: Configuration object
        optimizer: Optimizer instance
    
    Returns:
        Learning rate scheduler
    """
    import torch.optim as optim
    
    training_cfg = config.get('training', {})
    scheduler_cfg = training_cfg.get('scheduler', {})
    
    name = scheduler_cfg.get('name', None)
    if name is None:
        return None
    
    if name == 'cosine':
        epochs = training_cfg.get('epochs', 100)
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs,
            eta_min=scheduler_cfg.get('min_lr', 0)
        )
    elif name == 'step':
        step_size = scheduler_cfg.get('step_size', 30)
        gamma = scheduler_cfg.get('gamma', 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == 'multi_step':
        milestones = scheduler_cfg.get('milestones', [30, 60, 90])
        gamma = scheduler_cfg.get('gamma', 0.1)
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif name == 'plateau':
        patience = scheduler_cfg.get('patience', 10)
        factor = scheduler_cfg.get('factor', 0.1)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, verbose=True
        )
    
    raise ValueError(f"Unsupported scheduler: {name}")


def build_trainer(config, model, train_loader, val_loader, criterion, optimizer, scheduler, device, distributed):
    """
    Build trainer based on configuration.
    
    Args:
        config: Configuration object
        model: Model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss criterion
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device
        distributed: Whether using distributed training
    
    Returns:
        Trainer instance
    """
    model_cfg = config.get('model', {})
    model_type = model_cfg.get('type', 'classification')
    output_dir = config.get('output_dir', './output')
    
    if model_type == 'classification' or model_type == 'hateful_memes':
        return ClassificationTrainer(
            num_classes=model_cfg.get('num_classes', 2),
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            distributed=distributed,
            output_dir=output_dir
        )
    elif model_type == 'regression':
        return RegressionTrainer(
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            distributed=distributed,
            output_dir=output_dir
        )
    elif model_type == 'detection':
        training_cfg = config.get('training', {})
        eval_settings = training_cfg.get('eval_settings', {})
        
        return DetectionTrainer(
            iou_threshold=eval_settings.get('iou_threshold', 0.5),
            score_threshold=eval_settings.get('score_threshold', 0.05),
            model=model,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            distributed=distributed,
            output_dir=output_dir
        )
    
    raise ValueError(f"Unsupported model type: {model_type}")


def main_worker(rank, world_size, args, config):
    """
    Main worker function for distributed training.
    
    Args:
        rank: Process rank
        world_size: Number of processes
        args: Command line arguments
        config: Configuration object
    """
    # Set up distributed training
    if args.local_rank != -1:
        setup_distributed(backend='nccl')
    
    # Set the device
    if rank == 0 or args.local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Set random seed for reproducibility
    set_seed(config.get('seed', 42) + rank)
    
    # Build model
    model = build_model(config)
    model = model.to(device)
    
    # Build datasets and dataloaders
    train_dataset, val_dataset, test_dataset = build_datasets(config)
    train_loader, val_loader, test_loader = build_dataloaders(
        config, train_dataset, val_dataset, test_dataset, 
        distributed=(args.local_rank != -1)
    )
    
    # Build training components
    criterion = build_criterion(config)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    
    # Build trainer
    trainer = build_trainer(
        config, model, train_loader, val_loader, criterion, optimizer, scheduler, 
        device, distributed=(args.local_rank != -1)
    )
    
    # Resume from checkpoint if specified in args or config
    resume_config = config.get('resume', {})
    resume_enabled = resume_config.get('enabled', False)
    checkpoint_path = args.resume or (resume_enabled and resume_config.get('checkpoint_path', None))
    
    if checkpoint_path:
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(
            checkpoint_path,
            strict=resume_config.get('strict', True),
            reset_optimizer=resume_config.get('reset_optimizer', False),
            reset_lr_scheduler=resume_config.get('reset_lr_scheduler', False),
            reset_epoch=resume_config.get('reset_epoch', False)
        )
    
    # Start training
    epochs = config.get('training', {}).get('epochs', 100)
    trainer.train(epochs)
    
    # Save final model
    if rank == 0 or args.local_rank == -1:
        final_checkpoint_path = os.path.join(config.get('output_dir', './output'), 'final_model.pth')
        model_to_save = model.module if hasattr(model, 'module') else model
        
        torch.save({
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config
        }, final_checkpoint_path)
        
        print(f"Training completed. Final model saved to {final_checkpoint_path}")


def main():
    """
    Main function to start training.
    """
    args = parse_args()
    
    # Load configuration
    config = Config(args.config)
    
    # Create output directory
    output_dir = config.get('output_dir', './output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save a copy of the config
    config_save_path = os.path.join(output_dir, 'config.yaml')
    config.save(config_save_path)
    
    # Get distributed training settings
    distributed_cfg = config.get('distributed', {})
    use_distributed = distributed_cfg.get('enabled', False) and torch.cuda.device_count() > 1
    
    if use_distributed and args.local_rank == -1:
        # Launch with torch.distributed.launch
        world_size = torch.cuda.device_count()
        mp.spawn(main_worker, args=(world_size, args, config), nprocs=world_size)
    else:
        # Single GPU or launched with torch.distributed.launch
        main_worker(args.local_rank, 1, args, config)


if __name__ == "__main__":
    main()