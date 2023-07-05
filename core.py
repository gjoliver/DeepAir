"""
A minimal job showing how to Fine-tune gpt2 with tiny_shakespeare dataset
using DeepSpeed and Ray core library directly.

Note that you won't get fault tolerance with this approach.
"""

import deepspeed
import ray
import torch

from util import (
    collate_fn,
    deepspeed_config,
    get_datasets,
    get_model,
    loss_fn,
    to_device,
)


BATCH_SIZE = 8
NUM_WORKERS = 4


@ray.remote(num_gpus=1)
def train_loop_per_worker(world_size: int, rank: int):
    assert torch.cuda.is_available(), "Example workload only works with GPUs!"
    assert BATCH_SIZE % world_size == 0, "Batch size must be divisible by world size!"

    model, tokenizer = get_model()

    ds = get_datasets(tokenizer, world_size, rank)
    data_loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, collate_fn=collate_fn,
    )

    model, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        # DeepSpeed config.
        config=deepspeed_config(BATCH_SIZE / world_size),
        # DeepSpeed initializes Torch DDP process group.
        dist_init_required=True,
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

        if rank == 0:
            # Print stats from Rank 0 worker.
            print(f"step: {step}, loss: {torch.mean(loss).detach().cpu().numpy()}")


def train():
    """Main entry point."""
    ray.get([
        train_loop_per_worker.remote(NUM_WORKERS, i) for i in range(NUM_WORKERS)
    ])
    print("done!")


if __name__ == "__main__":
    train()
