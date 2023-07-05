"""
A minimal job showing how to Fine-tune gpt2 with tiny_shakespeare dataset
using DeepSpeed and Ray core library directly.

Note that you won't get fault tolerance that is usually provided by Ray AIR.

Also, this script only works with single-machine multi-gpu case.
If your cluster has multiple GPU instances, try using TorchTrainer instead,
which properly sets up all the env vars (torch_trainer.py).
"""

from contextlib import closing
import os
import socket
from typing import Tuple

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


def find_ip_and_free_port() -> Tuple[str, int]:
    ip = ray.util.get_node_ip_address()
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
    return ip, port


def set_env_vars(master_ip: str, master_port: int, world_size: int, rank: int):
    os.environ['DS_ACCELERATOR'] = 'cuda'

    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    # Note: this assumes a single instance with multiple GPU cards.
    # Local rank will have to be computed if we want this script to
    # work with multiple GPU instances in this cluster.
    # TODO: make it work for multi-instance case.
    os.environ['LOCAL_RANK'] = str(rank)

    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = str(master_port)


@ray.remote(num_gpus=1)
def train_loop_per_worker(
    master_ip: str,
    master_port: int,
    world_size: int,
    rank: int
):
    assert torch.cuda.is_available(), "Example workload only works with GPUs!"
    assert BATCH_SIZE % world_size == 0, "Batch size must be divisible by world size!"
    per_gpu_batch_size = int(BATCH_SIZE / world_size)

    model, tokenizer = get_model()

    ds = get_datasets(tokenizer, world_size, rank)
    data_loader = torch.utils.data.DataLoader(
        ds, batch_size=per_gpu_batch_size, collate_fn=collate_fn,
    )

    # Must do before DeepSpeed initialization.
    set_env_vars(master_ip, master_port, world_size, rank)

    model, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        # DeepSpeed config.
        config=deepspeed_config(per_gpu_batch_size),
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
    master_ip, master_port = find_ip_and_free_port()

    ray.get([
        train_loop_per_worker.remote(
            master_ip, master_port, NUM_WORKERS, i
        ) for i in range(NUM_WORKERS)
    ])

    print("done!")


if __name__ == "__main__":
    train()
