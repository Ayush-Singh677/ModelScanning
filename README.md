# Model Scanning and Security Project

This project contains tools for model security analysis, including serialization attack detection, membership inference attacks, and adversarial example generation.

## Setup

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `serialisation/` - Tools for detecting malicious model serialization
    - `scanner.py` - Model file scanner
    - `inject.py` - Pickle injection utilities
    - `helper.ipynb` - Helper notebook for model generation

- `mia/` - Membership Inference Attack implementation
    - `cifar10.py` - MIA example on CIFAR10
    - `utils.py` - Shadow model utilities

- `adversarial/` - Adversarial example generation and detection
    - `adv_pgd.py` - PGD attack implementation
    - `imagenet_classes.txt` - ImageNet class labels

## Usage

### Serialization Attack Detection

```
python scanner.py /path/to/model.pt
```

### Membership Inference Attack

```
python mia/cifar10.py --target_epochs 12 --attack_epochs 6 --num_shadows 3
```

### Adversarial Example Generation

```
python adversarial/adv_pgd.py
```

## Requirements

See `requirements.txt` for full list of dependencies. Key requirements:

- PyTorch
- TensorFlow 
- scikit-learn
- numpy
- matplotlib
- tqdm