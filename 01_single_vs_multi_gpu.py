# 01_single_vs_multi_gpu.py
"""
So sánh hiệu suất training giữa single GPU và multi-GPU
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import time
from utils import TrainingMetrics, print_gpu_memory_usage

def train_single_gpu(model_name="facebook/opt-iml-1.3b", batch_size=2, max_steps=50):
    print("\n=== Single GPU Training ===")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    metrics = TrainingMetrics()
    metrics.start_training()
    model.train()
    for step in range(max_steps):
        input_ids = torch.randint(0, 50257, (batch_size, 512)).to(model.device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()
        optimizer.step()
        metrics.log_metrics(loss.item(), step)
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
    print(f"Training completed in {metrics.get_summary()['total_time']:.2f} seconds")
    return metrics

def train_multi_gpu(model_name="facebook/opt-iml-1.3b", batch_size=4, max_steps=50):
    print("\n=== Multi-GPU Training ===")
    if torch.cuda.device_count() < 2:
        print("Not enough GPUs for multi-GPU training")
        return None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    metrics = TrainingMetrics()
    metrics.start_training()
    model.train()
    for step in range(max_steps):
        input_ids = torch.randint(0, 50257, (batch_size, 512)).to(model.device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()
        optimizer.step()
        metrics.log_metrics(loss.item(), step)
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
    print(f"Training completed in {metrics.get_summary()['total_time']:.2f} seconds")
    return metrics

if __name__ == "__main__":
    single_gpu_metrics = train_single_gpu()
    multi_gpu_metrics = train_multi_gpu()
    if single_gpu_metrics and multi_gpu_metrics:
        print("\n=== Comparison ===")
        single_summary = single_gpu_metrics.get_summary()
        multi_summary = multi_gpu_metrics.get_summary()
        print(f"Single GPU:\n  Final Loss: {single_summary['final_loss']:.4f}\n  Total Time: {single_summary['total_time']:.2f}s\n  Steps/Second: {single_summary['steps_per_second']:.2f}")
        print(f"\nMulti-GPU:\n  Final Loss: {multi_summary['final_loss']:.4f}\n  Total Time: {multi_summary['total_time']:.2f}s\n  Steps/Second: {multi_summary['steps_per_second']:.2f}")
        speedup = multi_summary['steps_per_second'] / single_summary['steps_per_second']
        print(f"\nSpeedup: {speedup:.2f}x")
        single_gpu_metrics.save_metrics('single_gpu_metrics.json')
        multi_gpu_metrics.save_metrics('multi_gpu_metrics.json') 