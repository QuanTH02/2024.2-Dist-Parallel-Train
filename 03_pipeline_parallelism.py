# 03_pipeline_parallelism.py
"""
So sánh DeepSpeed Pipeline vs PyTorch Pipeline
"""
import torch
import torch.nn as nn
from torch.distributed import pipeline
from deepspeed.pipe import PipelineModule, LayerSpec
import deepspeed
import json
from utils import TrainingMetrics

def train_pytorch_pipeline(batch_size=4, max_steps=50, num_stages=2):
    print(f"\n=== PyTorch Pipeline ({num_stages} stages) ===")
    if torch.cuda.device_count() < num_stages:
        print(f"Need {num_stages} GPUs. Skipping...")
        return None
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=50257, hidden_size=768, num_layers=6):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(hidden_size, 8, batch_first=True)
                for _ in range(num_layers)
            ])
            self.output = nn.Linear(hidden_size, vocab_size)
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x)
            return self.output(x)
    model = SimpleModel()
    layers_per_stage = len(model.layers) // num_stages
    stages = []
    stage0 = nn.Sequential(model.embedding, *model.layers[:layers_per_stage])
    stages.append(stage0)
    for i in range(1, num_stages - 1):
        start = i * layers_per_stage
        end = (i + 1) * layers_per_stage
        stage = nn.Sequential(*model.layers[start:end])
        stages.append(stage)
    final_stage = nn.Sequential(*model.layers[(num_stages - 1) * layers_per_stage:], model.output)
    stages.append(final_stage)
    pipe = pipeline.Pipeline(stages, chunks=1, num_stages=num_stages)
    optimizer = torch.optim.AdamW(pipe.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()
    metrics = TrainingMetrics()
    metrics.start_training()
    pipe.train()
    for step in range(max_steps):
        optimizer.zero_grad()
        input_ids = torch.randint(0, 50257, (batch_size, 512))
        labels = torch.randint(0, 50257, (batch_size, 512))
        output = pipe(input_ids)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss.backward()
        optimizer.step()
        metrics.log_metrics(loss.item(), step)
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
    print(f"Training completed in {metrics.get_summary()['total_time']:.2f} seconds")
    return metrics

def train_deepspeed_pipeline(batch_size=4, max_steps=50, num_stages=2):
    print(f"\n=== DeepSpeed Pipeline ({num_stages} stages) ===")
    if torch.cuda.device_count() < num_stages:
        print(f"Need {num_stages} GPUs. Skipping...")
        return None
    ds_config = {
        "train_batch_size": batch_size,
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 5e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            }
        },
        "fp16": {"enabled": True},
        "pipeline": {"stages": num_stages},
        "steps_per_print": 10
    }
    config_path = f"ds_pipeline_{num_stages}stages.json"
    with open(config_path, 'w') as f:
        json.dump(ds_config, f, indent=2)
    specs = [
        LayerSpec(nn.Embedding, 50257, 768),
        LayerSpec(nn.TransformerEncoderLayer, 768, 8, batch_first=True),
        LayerSpec(nn.TransformerEncoderLayer, 768, 8, batch_first=True),
        LayerSpec(nn.TransformerEncoderLayer, 768, 8, batch_first=True),
        LayerSpec(nn.Linear, 768, 50257)
    ]
    model = PipelineModule(layers=specs, loss_fn=nn.CrossEntropyLoss(), num_stages=num_stages)
    model_engine, optimizer, _, _ = deepspeed.initialize(model=model, config=config_path)
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

if __name__ == "__main__":
    pytorch_pipeline = train_pytorch_pipeline(num_stages=2)
    deepspeed_pipeline = train_deepspeed_pipeline(num_stages=2)
    all_metrics = {}
    if pytorch_pipeline:
        all_metrics['PyTorch Pipeline'] = pytorch_pipeline
    if deepspeed_pipeline:
        all_metrics['DeepSpeed Pipeline'] = deepspeed_pipeline
    if all_metrics:
        print("\n=== Comparison ===")
        for name, metrics in all_metrics.items():
            summary = metrics.get_summary()
            print(f"\n{name}:\n  Final Loss: {summary['final_loss']:.4f}\n  Total Time: {summary['total_time']:.2f}s\n  Steps/Second: {summary['steps_per_second']:.2f}")
            safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").replace("/", "_")
            metrics.save_metrics(f'{safe_name}_metrics.json')
        
    else:
        print("No pipeline training results available.") 