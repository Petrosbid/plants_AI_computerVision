# Plant Disease Detection with PyTorch

A computer vision project for plant disease detection using PyTorch and transfer learning with EfficientNetV2.

## Features

- PyTorch-based implementation for better Windows GPU support
- Transfer learning with EfficientNetV2 backbone
- Two-phase training: Head training followed by fine-tuning
- MLflow integration for experiment tracking
- Data augmentation and preprocessing
- GPU acceleration support

## Requirements

- Python 3.7+
- PyTorch with CUDA support (for GPU training)
- MLflow for experiment tracking

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   For GPU support:
   ```bash
   pip install -r requirements-gpu.txt
   ```

## Usage

1. Prepare your dataset in the `data/` directory with subdirectories for each class
2. Update `config/params.yaml` with your dataset path and training parameters
3. Run training:
   ```bash
   python train.py
   ```

## Check GPU Availability

To check if your GPU is properly detected:

```bash
python check_gpu.py
```