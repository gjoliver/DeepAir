from typing import Tuple

from datasets import Dataset, load_dataset
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def to_device(batch):
    output = {}
    for k, v in batch.items():
        output[k] = v.to(torch.device("cuda"))
    return output


def loss_fn(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    labels = labels.contiguous().long()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


def collate_fn(batch):
    return {
        key: torch.stack([torch.tensor(r[key]) for r in batch])
        for key in ["input_ids", "attention_mask", "labels"]
    }
