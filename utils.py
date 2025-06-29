import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import numpy as np
import time
import json
import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

def setup_distributed():
    """Setup distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0

def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" if device == "cuda" else None
    )
    
    return model, tokenizer

def load_wikitext_dataset(tokenizer, max_length: int = 512):
    """Load and preprocess WikiText dataset"""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def create_dataloader(dataset, batch_size: int, shuffle: bool = True):
    """Create DataLoader for the dataset"""
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=None,  # Will be set in training
        mlm=False
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collator
    )

class TrainingMetrics:
    """Class to track training metrics"""
    def __init__(self):
        self.losses = []
        self.times = []
        self.start_time = None
    
    def start_training(self):
        self.start_time = time.time()
    
    def log_metrics(self, loss: float, step: int):
        current_time = time.time()
        self.losses.append(loss)
        self.times.append(current_time - self.start_time)
    
    def get_summary(self) -> Dict:
        return {
            'final_loss': self.losses[-1] if self.losses else None,
            'avg_loss': np.mean(self.losses) if self.losses else None,
            'total_time': self.times[-1] if self.times else None,
            'steps_per_second': len(self.losses) / self.times[-1] if self.times else None
        }
    
    def save_metrics(self, filename: str):
        metrics = {
            'losses': self.losses,
            'times': self.times,
            'summary': self.get_summary()
        }
        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)

def plot_training_curves(metrics_dict: Dict[str, TrainingMetrics], save_path: str = None):
    """Plot training curves for comparison"""
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    for name, metrics in metrics_dict.items():
        plt.plot(metrics.losses, label=name)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Time curves
    plt.subplot(1, 3, 2)
    for name, metrics in metrics_dict.items():
        plt.plot(metrics.times, metrics.losses, label=name)
    plt.title('Loss vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Performance comparison
    plt.subplot(1, 3, 3)
    names = list(metrics_dict.keys())
    final_losses = [metrics.get_summary()['final_loss'] for metrics in metrics_dict.values()]
    total_times = [metrics.get_summary()['total_time'] for metrics in metrics_dict.values()]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width/2, final_losses, width, label='Final Loss', alpha=0.8)
    bars2 = ax2.bar(x + width/2, total_times, width, label='Total Time (s)', alpha=0.8)
    
    ax1.set_xlabel('Methods')
    ax1.set_ylabel('Final Loss')
    ax2.set_ylabel('Total Time (s)')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def print_gpu_memory_usage():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    else:
        print("No GPU available")

def create_deepspeed_config(zero_stage: int = 3, offload_optimizer: bool = False, offload_param: bool = False):
    """Create DeepSpeed configuration"""
    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto"
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto"
            }
        },
        "fp16": {
            "enabled": "auto"
        },
        "zero_optimization": {
            "stage": zero_stage,
            "offload_optimizer": {
                "device": "cpu" if offload_optimizer else "none",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu" if offload_param else "none",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "gradient_clipping": "auto",
        "steps_per_print": 10,
        "wall_clock_breakdown": False
    }
    
    return config 