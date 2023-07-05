"""
A minimal job showing how to Fine-tune gpt2 with tiny_shakespeare dataset
using DeepSpeed and Ray AIR's TorchTrainer.
"""

import os
from typing import Any, Dict

import deepspeed
from ray.air import session
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
import torch

from util import (
    collate_fn,
    deepspeed_config,
    get_datasets,
    get_model,
    loss_fn,
    to_device,
)

BATCH_SIZE = 16
NUM_WORKERS = 4


def train_loop_per_worker(config: Dict[str, Any]):
    assert torch.cuda.is_available(), "Example workload only works with GPUs!"
    assert BATCH_SIZE % session.get_world_size() == 0, (
        "Batch size must be divisible by world size!"
    )
    per_gpu_batch_size = int(BATCH_SIZE / session.get_world_size())

    model, tokenizer = get_model()

    ds = get_datasets(
        tokenizer,
        session.get_world_size(),
        session.get_world_rank(),
    )
    data_loader = torch.utils.data.DataLoader(
        ds, batch_size=per_gpu_batch_size, collate_fn=collate_fn,
    )

    model, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        # DeepSpeed config.
        config=deepspeed_config(per_gpu_batch_size),
        # TorchTrainer handled this.
        dist_init_required=False,
    )

    # For demo purpose, we only train 1 epoch.
    for step, batch in enumerate(data_loader):
        batch = to_device(batch)
        outputs = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        loss = loss_fn(outputs["logits"], batch["labels"])

        # DeepSpeed engine handles backward and step.
        model.backward(loss)
        model.step()
        model.zero_grad()

        session.report({
            "step": step,
            "loss": torch.mean(loss).detach().cpu().numpy(),
        })


def train():
    """Main entry point."""
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=ScalingConfig(
            num_workers=NUM_WORKERS, use_gpu=True
        ),
    )

    result = trainer.fit()
    print(result)


if __name__ == "__main__":
    train()
