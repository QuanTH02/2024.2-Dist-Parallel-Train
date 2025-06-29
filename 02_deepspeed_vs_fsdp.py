# 02_deepspeed_vs_fsdp.py
"""
So sánh hiệu suất giữa DeepSpeed ZeRO và PyTorch FSDP
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
import json
from utils import TrainingMetrics, create_deepspeed_config

def train_deepspeed_zero(model_name="facebook/opt-iml-1.3b", zero_stage=3, batch_size=4, max_steps=50):
    print(f"\n=== DeepSpeed ZeRO Stage {zero_stage} ===")
    ds_config = create_deepspeed_config(zero_stage=zero_stage)
    config_path = f"ds_config_zero{zero_stage}.json"
    with open(config_path, 'w') as f:
        json.dump(ds_config, f, indent=2)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, config=config_path, model_parameters=model.parameters()
    )
    metrics = TrainingMetrics()
    metrics.start_training()
    model_engine.train()
    for step in range(max_steps):
        input_ids = torch.randint(0, 50257, (batch_size, 512)).to(model_engine.device)
        labels = torch.randint(0, 50257, (batch_size, 512)).to(model_engine.device)
        loss = model_engine(input_ids, labels)
        model_engine.backward(loss)
        model_engine.step()
        metrics.log_metrics(loss.item(), step)
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
    print(f"Training completed in {metrics.get_summary()['total_time']:.2f} seconds")
    return metrics

def train_fsdp(model_name="facebook/opt-iml-1.3b", batch_size=4, max_steps=50, cpu_offload=False):
    print(f"\n=== PyTorch FSDP (CPU Offload: {cpu_offload}) ===")
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
    from torch.distributed.fsdp.wrap import default_auto_wrap_policy
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    fsdp_model = FSDP(
        model,
        fsdp_auto_wrap_policy=default_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=cpu_offload)
    )
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    metrics = TrainingMetrics()
    metrics.start_training()
    fsdp_model.train()
    for step in range(max_steps):
        input_ids = torch.randint(0, 50257, (batch_size, 512)).to(fsdp_model.device)
        optimizer.zero_grad()
        outputs = fsdp_model(input_ids=input_ids)
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
    deepspeed_zero3 = train_deepspeed_zero(zero_stage=3)
    fsdp_no_offload = train_fsdp(cpu_offload=False)
    fsdp_with_offload = train_fsdp(cpu_offload=True)
    all_metrics = {}
    if deepspeed_zero3:
        all_metrics['DeepSpeed ZeRO-3'] = deepspeed_zero3
    if fsdp_no_offload:
        all_metrics['PyTorch FSDP'] = fsdp_no_offload
    if fsdp_with_offload:
        all_metrics['PyTorch FSDP (CPU Offload)'] = fsdp_with_offload
    print("\n=== Comparison ===")
    for name, metrics in all_metrics.items():
        summary = metrics.get_summary()
        print(f"\n{name}:\n  Final Loss: {summary['final_loss']:.4f}\n  Total Time: {summary['total_time']:.2f}s\n  Steps/Second: {summary['steps_per_second']:.2f}")
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").replace("/", "_")
        metrics.save_metrics(f'{safe_name}_metrics.json') 