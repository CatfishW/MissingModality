# MissingModality

A flexible deep learning framework for training with full modalities and testing with missing modalities.

## Features

- Support for multiple input modalities
- Graceful handling of missing modalities during inference
- Compatible with various task types (classification, detection, regression)
- Distributed training across multiple GPUs
- Tensorboard integration for visualization and monitoring
- Modular design for easy extension

## Project Structure

```
MissingModality/
├── configs/                # Configuration files for different tasks
│   ├── classification_example.yaml  # Example config for classification
│   └── detection_example.yaml       # Example config for detection
├── data/                   # Data handling modules
│   ├── datasets/           # Dataset implementations
│   │   └── base.py         # Base classes for multi-modal datasets
│   ├── loaders/            # DataLoader utilities
│   │   └── dataloader.py   # Distributed training compatible loaders
│   └── transforms/         # Data transformation utilities
├── models/                 # Model implementations
│   ├── backbones/          # Feature extractors for different modalities
│   ├── fusion/             # Modality fusion strategies
│   │   └── fusion_modules.py  # Different fusion approaches
│   ├── heads/              # Task-specific heads
│   └── missing/            # Missing modality handling
│       └── base_model.py   # Base model with missing modality support
├── trainers/               # Training implementations
│   ├── base_trainer.py     # Base trainer with distributed support
│   └── task_trainers.py    # Specialized trainers for different tasks
├── utils/                  # Utility functions
│   ├── config.py           # Configuration handling
│   └── distributed.py      # Distributed training utilities
├── visualization/          # Visualization tools
│   └── tensorboard.py      # TensorBoard integration
├── main.py                 # Main entry point
├── train.py                # Training script
├── test.py                 # Testing script with missing modality evaluation
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MissingModality.git
cd MissingModality

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train.py --config configs/example_config.yaml
```

### Testing

```bash
python test.py --config configs/example_config.yaml --checkpoint checkpoints/model.pth
```

## License

MIT