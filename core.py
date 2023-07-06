"""
A minimal job showing how to Fine-tune gpt2 with tiny_shakespeare dataset
using DeepSpeed and Ray core library directly.

Note that you won't get fault tolerance that is usually provided by Ray AIR.
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


def set_env_vars(master_addr: str, master_port: int, world_size: int, rank: int):
    os.environ['DS_ACCELERATOR'] = 'cuda'
    os.environ['NCCL_SOCKET_IFNAME'] = '^lo,docker,veth'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'

    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    # Note: this only works for the case where there is 1 GPU per machine instance.
    # TODO: make this example work for arbitrary instance & GPU setups.
    os.environ['LOCAL_RANK'] = '0'

    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)


@ray.remote(num_gpus=1)
class Trainer:
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank

    def find_ip_and_free_port(self) -> Tuple[str, int]:
        ip = ray.util.get_node_ip_address()
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
        return ip, port

    def initilize(self, master_addr: str, master_port: int):
        assert torch.cuda.is_available(), "Example workload only works with GPUs!"
        assert BATCH_SIZE % self.world_size == 0, "Batch size must be divisible by world size!"

        per_gpu_batch_size = int(BATCH_SIZE / self.world_size)

        model, tokenizer = get_model()

        ds = get_datasets(tokenizer, self.world_size, self.rank)
        data_loader = torch.utils.data.DataLoader(
            ds, batch_size=per_gpu_batch_size, collate_fn=collate_fn,
        )

        # Must do before DeepSpeed initialization.
        set_env_vars(master_addr, master_port, self.world_size, self.rank)
        model, _, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            # DeepSpeed config.
            config=deepspeed_config(per_gpu_batch_size),
            # DeepSpeed initializes Torch DDP process group.
            dist_init_required=True,
        )

        self.tokenizer = tokenizer
        self.model = model
        self.data_loader = data_loader

    def train_loop(self):
        # For demo purpose, we only train 1 epoch.
        for step, batch in enumerate(self.data_loader):
            batch = to_device(batch)
            outputs = self.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )

            loss = loss_fn(outputs["logits"], batch["labels"])

            # DeepSpeed engine handles backward and step.
            self.model.backward(loss)
            self.model.step()
            self.model.zero_grad()

            # Print stats from Rank 0 worker.
            print(f"step: {step}, loss: {torch.mean(loss).detach().cpu().numpy()}")


def train():
    """Main entry point."""
    trainers = [Trainer.remote(NUM_WORKERS, i) for i in range(NUM_WORKERS)]

    master_addr, master_port = ray.get(
        trainers[0].find_ip_and_free_port.remote()
    )
    ray.get([
        trainer.initilize.remote(master_addr, master_port) for trainer in trainers
    ])

    ray.get([trainer.train_loop.remote() for trainer in trainers])

    print("done!")


if __name__ == "__main__":
    train()
