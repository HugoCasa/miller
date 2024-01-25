from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import torch
import yaml
import random
from tqdm import tqdm

from .prompts import prepare_prompt, prepare_label


class SimpleDataset(Dataset):
    def __init__(
        self,
        input_sequences: list[str],
        labels: list[str],
        metadata: list[dict],
    ):
        self.data = [
            {"input_sequences": i, "labels": l, "metadata": m}
            for i, l, m in zip(input_sequences, labels, metadata)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PPODataset(Dataset):
    def __init__(
        self,
        input_sequences: list[str],
        metadata: list[dict],
    ):
        self.data = [
            {"input_sequences": i, "metadata": m}
            for i, m in zip(input_sequences, metadata)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DPODataset(Dataset):
    def __init__(
        self,
        input_sequences: list[str],
        chosen: list[str],
        rejected: list[str],
    ):
        self.data = [
            {"input_sequences": i, "chosen": c, "rejected": r}
            for i, c, r in zip(input_sequences, chosen, rejected)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_cond(batch, tokenizer):
    # pad to longest
    inputs = tokenizer(
        [b["input_sequences"] for b in batch],
        padding="longest",
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs["labels"] = tokenizer(
        [b["labels"] for b in batch],
        padding="longest",
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )["input_ids"]
    inputs["metadata"] = [b["metadata"] for b in batch]
    return inputs


def collate_fn_causal(batch, tokenizer: PreTrainedTokenizer):
    # pad to longest
    inputs = tokenizer(
        [b["input_sequences"] + b["labels"] for b in batch],
        padding="longest",
        return_tensors="pt",
        truncation=True,
        max_length=1216 - 1,  # eos is not added by default
    )
    # append eos token to input_ids tensors along dim=1
    inputs["input_ids"] = torch.cat(
        [
            inputs["input_ids"],
            torch.ones(len(batch), 1, dtype=torch.int64) * tokenizer.eos_token_id,
        ],
        dim=1,
    )
    inputs["attention_mask"] = torch.cat(
        [inputs["attention_mask"], torch.ones(len(batch), 1, dtype=torch.int64)], dim=1
    )

    labels = inputs["input_ids"].clone()
    # labels to show (label + eos)
    label_lens = [1 + len(tokenizer.tokenize(b["labels"])) for b in batch]

    for i, l in enumerate(label_lens):
        # set everything before labels to -100 (left padding)
        labels[i, :-l] = -100

    inputs["labels"] = labels

    inputs["metadata"] = [b["metadata"] for b in batch]
    return inputs


def get_datasets(model_name, enable_openapi_rag=False):
    with open("data/processed/wm_dataset_v2.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    with open("data/synthetic_data/hubfix_generated.yaml", "r") as f:
        hubfix = yaml.load(f, Loader=yaml.FullLoader)
    with open("data/synthetic_data/hubedit_generated.yaml", "r") as f:
        hubedit = yaml.load(f, Loader=yaml.FullLoader)
    with open("data/processed/benchmark_dataset_v2.yaml", "r") as f:
        benchmark = yaml.load(f, Loader=yaml.FullLoader)
        benchmark_fix_ids = ["hubfix_" + d["id"] for d in benchmark]
        benchmark_edit_ids = ["hubedit_" + d["id"] for d in benchmark]
    if enable_openapi_rag:
        with open("data/openapi/scripts.yaml", "r") as f:
            openapi_scripts = yaml.load(f, Loader=yaml.FullLoader)
        with open("data/openapi/scripts_py.yaml", "r") as f:
            openapi_scripts += yaml.load(f, Loader=yaml.FullLoader)

    hubfix = [d for d in hubfix if d["id"] not in benchmark_fix_ids]
    hubedit = [d for d in hubedit if d["id"] not in benchmark_edit_ids]
    data = data + hubfix + hubedit

    if enable_openapi_rag:
        data = data + openapi_scripts

    random.seed(42)
    random.shuffle(data)
    print("Preparing data prompts...")
    input_sequences = [
        prepare_prompt(
            d,
            model_name=model_name,
            openapi_scripts=openapi_scripts if enable_openapi_rag else None,
        )
        for d in tqdm(data)
    ]

    labels = [prepare_label(d) for d in data]
    metadata = [{**d, "openapi_info": None} for d in data]

    gen_index = [
        i
        for i, d in enumerate(data)
        if not d["id"].startswith("hubfix_")
        and not d["id"].startswith("hubedit_")
        and not d["id"].startswith("openapi_")
    ][0]
    print("Example prompt with label (gen):")
    print(input_sequences[gen_index] + labels[gen_index])

    if enable_openapi_rag:
        openapi_index = [
            i for i, d in enumerate(data) if d["id"].startswith("openapi_")
        ][0]
        print("Example prompt with label (OpenAPI):")
        print(input_sequences[openapi_index] + labels[openapi_index])

    train_set_size = int(len(data) * 0.8)
    train_dataset = SimpleDataset(
        input_sequences[:train_set_size],
        labels[:train_set_size],
        metadata[:train_set_size],
    )
    val_dataset = SimpleDataset(
        input_sequences[train_set_size:],
        labels[train_set_size:],
        metadata[train_set_size:],
    )

    return train_dataset, val_dataset
