#!/usr/bin/env python3
"""
Setup script cho Kaggle environment
"""

import os
import subprocess
import sys

def setup_kaggle():
    """Setup environment cho Kaggle"""
    print("🚀 Setting up Kaggle environment...")
    
    # Cài đặt dependencies
    packages = [
        "torch>=2.0.0",
        "transformers>=4.30.0", 
        "datasets>=2.12.0",
        "deepspeed>=0.10.0",
        "accelerate>=0.20.0",
        "wandb>=0.15.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "jupyter",
        "ipywidgets"
    ]
    
    for package in packages:
        print(f"📦 Installing {package}...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ], check=True, capture_output=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {package}")
    
    # Kiểm tra GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ Found {gpu_count} GPU(s)")
            for i in range(gpu_count):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("⚠️  No GPU available")
    except ImportError:
        print("❌ PyTorch not available")
    
    # Tạo thư mục output
    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("✅ Kaggle setup completed!")

if __name__ == "__main__":
    setup_kaggle() 