"""
For demo purpose, a minimal job showing how to Fine-tune gpt2
using tiny_shakespeare dataset with Ray AIR's TorchTrainer.
"""

from typing import Any, Dict, Tuple

from datasets import load_dataset
import deepspeed
import ray
from ray.air import session
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.air.config import RunConfig
from ray.air.config import CheckpointConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BATCH_SIZE = 16


def get_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    MODEL = "gpt2"

    model = AutoModelForCausalLM.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_datasets(
    tokenizer: AutoTokenizer, world_size: int, rank: int
) -> ray.data.Dataset:
    """Download and pre-process the datasets."""
    datasets = load_dataset("tiny_shakespeare")

    def split(rows):
        lines = []
        for row in rows["text"]: lines.extend(row.split("\n"))
        return {"text": lines}

    def tokenize(row):
        return tokenizer(
            row["text"],
            max_length=tokenizer.model_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )

    def preprocess(ds):
        return ds.map(
            split, batched=True,
        ).shard(
            num_shards=world_size, index=rank
        ).map(
            tokenize, remove_columns=["text"]
        )

    return preprocess(datasets["train"])


def _to_device(batch):
    output = {}
    for k, v in batch.items():
        output[k] = v.to(torch.device("cuda"))
    return output


def loss_fn(logits, batch):
    pass


def train_loop_per_worker(config: Dict[str, Any]):
    assert torch.cuda.is_available(), "Example workload only works with GPUs!"

    model, tokenizer = get_model()

    train_dataset = get_datasets(
        tokenizer,
        session.get_world_size(),
        session.get_world_rank(),
    )
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)

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
            "bf16": {
                "enabled": True
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
    for batch in data_loader:
        batch = _to_device(batch)
        outputs = model(
            input_ids=batch["input_ids"],
            # Trick to make sure attention mask is bool.
            attention_mask=(batch["attention_mask"] > 0),
        )

        loss = loss_fn(outputs["logits"], batch)

        # DeepSpeed engine handles backward and step.
        model.backward(loss)
        model.step()
        model.zero_grad()

        session.report({"loss": loss.cpu()})


def train():
    """Main entry point."""
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        scaling_config=ScalingConfig(num_workers=3, use_gpu=False),
        run_config=RunConfig(checkpoint_config=CheckpointConfig()),
    )

    result = trainer.fit()
    print(result)


if __name__ == "__main__":
    train()
