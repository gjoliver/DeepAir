"""
For demo purpose, a minimal job showing how to Fine-tune gpt2
using tiny_shakespeare dataset with Ray AIR's TorchTrainer.
"""

from typing import Any, Dict

import deepspeed
from ray.air import session
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
import torch

from util import collate_fn, get_datasets, get_model, loss_fn, to_device


BATCH_SIZE = 8
NUM_WORKERS = 4


def train_loop_per_worker(config: Dict[str, Any]):
    assert torch.cuda.is_available(), "Example workload only works with GPUs!"

    model, tokenizer = get_model()

    ds = get_datasets(
        tokenizer,
        session.get_world_size(),
        session.get_world_rank(),
    )
    data_loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, collate_fn=collate_fn,
    )

    model, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        # DeepSpeed config.
        config={
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-5,
                }
            },
            "fp16": {
                "enabled": True
            },
            "bf16": {
                # Turn this on if using AMPERE GPUs.
                "enabled": False
            },
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "none",
                },
                "offload_param": {
                    "device": "none",
                },
            },
            "gradient_accumulation_steps": 1,
            "gradient_clipping": True,
            "steps_per_print": 10,
            "train_micro_batch_size_per_gpu": BATCH_SIZE,
            "wall_clock_breakdown": False,
        },
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
