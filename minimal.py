"""
For demo purpose, a minimal job showing how to Fine-tune gpt2
using tiny_shakespeare dataset with Ray AIR's TorchTrainer.
"""

from typing import Any, Dict, Tuple

from datasets import Dataset, load_dataset
import deepspeed
import ray
from ray.air import session
from ray.train.torch import TorchTrainer
from ray.air.config import ScalingConfig
from ray.air.config import RunConfig
from ray.air.config import CheckpointConfig
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.deepspeed import HfDeepSpeedConfig


BATCH_SIZE = 8
NUM_WORKERS = 4
IGNORE_INDEX = -100


def get_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    MODEL = "gpt2"

    model = AutoModelForCausalLM.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def get_datasets(tokenizer: AutoTokenizer, world_size: int, rank: int) -> Dataset:
    """Download and pre-process the datasets."""
    datasets = load_dataset("tiny_shakespeare")

    def split(rows):
        lines = []
        for row in rows["text"]: lines.extend(
            [r for r in row.split("\n") if r]  # Skip empty strings.
        )
        return {"text": lines}

    def tokenize(row):
        inputs = tokenizer(
            row["text"],
            max_length=tokenizer.model_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )

        labels = inputs.input_ids.copy()[:, 1:]
        # We used eos_token as the pad token.
        labels[labels == tokenizer.eos_token_id] = IGNORE_INDEX

        return {
            "input_ids": inputs.input_ids,
            # Trick to make sure attention mask is bool.
            "attention_mask": (inputs.attention_mask > 0),
            "labels": labels,
        }

    def preprocess(ds):
        return ds.map(
            split, batched=True,
        ).shard(
            num_shards=world_size, index=rank
        ).map(
            tokenize, batched=True, remove_columns=["text"]
        )

    return preprocess(datasets["train"])


def _to_device(batch):
    output = {}
    for k, v in batch.items():
        output[k] = v.to(torch.device("cuda"))
    return output


def _loss_fn(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    labels = labels.contiguous().long()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def _collate_fn(batch):
    return {
        key: torch.stack([torch.tensor(r[key]) for r in batch])
        for key in ["input_ids", "attention_mask", "labels"]
    }


def train_loop_per_worker(config: Dict[str, Any]):
    assert torch.cuda.is_available(), "Example workload only works with GPUs!"

    model, tokenizer = get_model()

    ds = get_datasets(
        tokenizer,
        session.get_world_size(),
        session.get_world_rank(),
    )
    data_loader = torch.utils.data.DataLoader(
        ds, batch_size=BATCH_SIZE, collate_fn=_collate_fn,
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
        batch = _to_device(batch)
        outputs = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        loss = _loss_fn(outputs["logits"], batch["labels"])

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
