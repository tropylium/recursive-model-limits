#!/usr/bin/env python
"""
Quick verification script to check if the training setup is working correctly.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

def check_environment_variables():
    """Check if required environment variables are set."""
    print("Checking environment variables...")
    errors = []
    
    if "DATASET_DIR" not in os.environ:
        errors.append("❌ DATASET_DIR not set in environment")
    else:
        print(f"✓ DATASET_DIR: {os.environ['DATASET_DIR']}")
    
    if "WANDB_PROJECT_NAME" not in os.environ:
        errors.append("❌ WANDB_PROJECT_NAME not set in environment")
    else:
        print(f"✓ WANDB_PROJECT_NAME: {os.environ['WANDB_PROJECT_NAME']}")
    
    return errors

def check_imports():
    """Check if required packages can be imported."""
    print("\nChecking imports...")
    errors = []
    
    packages = [
        "torch",
        "torchvision",
        "hydra",
        "omegaconf",
        "wandb",
        "tqdm",
        "pydantic",
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            errors.append(f"❌ {package} not installed")
    
    return errors

def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    import torch
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠ CUDA not available (CPU training only)")
    
    return []

def check_model_loading():
    """Check if model can be loaded."""
    print("\nChecking model loading...")
    errors = []
    
    try:
        from src.models.loader import load_model_class
        from src.models.simple_cnn import SimpleCNNConfig
        
        ModelClass = load_model_class("simple_cnn@SimpleCNN")
        config = SimpleCNNConfig(
            num_classes=10,
            input_channels=3,
            input_width=32,
            num_conv_layers=3,
            num_fc_layers=2,
            fc_hidden_dim=256,
        )
        model = ModelClass(config)
        print(f"✓ Model loaded successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        errors.append(f"❌ Failed to load model: {e}")
    
    return errors

def check_config_files():
    """Check if config files exist."""
    print("\nChecking config files...")
    errors = []
    
    config_files = [
        "conf/config.yaml",
        "conf/arch/simple_cnn.yaml",
        "conf/dataset/cifar10.yaml",
        "conf/dataset/cifar100.yaml",
        "conf/training/default.yaml",
        "conf/scheduler/cosine.yaml",
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✓ {config_file}")
        else:
            errors.append(f"❌ {config_file} not found")
    
    return errors

def main():
    print("=" * 80)
    print("Training Setup Verification")
    print("=" * 80)
    
    all_errors = []
    
    all_errors.extend(check_environment_variables())
    all_errors.extend(check_imports())
    all_errors.extend(check_cuda())
    all_errors.extend(check_config_files())
    all_errors.extend(check_model_loading())
    
    print("\n" + "=" * 80)
    if all_errors:
        print("❌ Setup verification FAILED")
        print("\nErrors found:")
        for error in all_errors:
            print(f"  {error}")
        print("\nPlease fix the errors above before training.")
        sys.exit(1)
    else:
        print("✅ Setup verification PASSED")
        print("\nYou're ready to start training!")
        print("\nQuick start:")
        print("  Single GPU:  python main.py")
        print("  Multi GPU:   torchrun --standalone --nproc_per_node=auto main.py")
    print("=" * 80)

if __name__ == "__main__":
    main()

