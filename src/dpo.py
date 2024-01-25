import torch
import yaml
from transformers import (
    AutoTokenizer,
    T5ForSequenceClassification,
    TrainingArguments,
    AutoModelForCausalLM,
)
from peft import PeftModel
from trl import DPOTrainer, create_reference_model
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
import os
import random
from datasets import load_dataset, Dataset
from functools import partial

from .prompts import prepare_prompt, REWARD_PROMPT
from .datasets import DPODataset
from .models import get_model_and_tokenizer

# MODEL_NAME = "salesforce/codet5p-220m"
MODEL_NAME = "ise-uiuc/Magicoder-S-DS-6.7B"

# CHECKPOINT_PATH = "models/salesforce/codet5p-220m_wmv2_local_2e-05_10_rag/model_10.pt"
CHECKPOINT_PATH = "/scratch/izar/casademo/pdm/models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_rag/model_10.pt"

ENABLE_OPENAPI_RAG = True

DEVICE = torch.device("cuda")

BATCH_SIZE = 4

LR = 1e-5

EPOCHS = 5


# def collate_fn(batch, tokenizer):
#     # pad to longest
#     prompts = tokenizer(
#         [b["input_sequences"] for b in batch],
#         padding="longest",
#         return_tensors="pt",
#         truncation=True,
#         max_length=512,
#     )
#     chosen = tokenizer(
#         [b["chosen"] for b in batch],
#         padding="longest",
#         return_tensors="pt",
#         truncation=True,
#         max_length=512,
#     )
#     rejected = tokenizer(
#         [b["rejected"] for b in batch],
#         padding="longest",
#         return_tensors="pt",
#         truncation=True,
#         max_length=512,
#     )
#     inputs = {
#         "prompt_input_ids": prompts["input_ids"],
#         "prompt_attention_mask": prompts["attention_mask"],
#         "chosen_labels": chosen["input_ids"],
#         "rejected_labels": rejected["input_ids"],
#     }
#     return inputs


def run_dpo(enable_openapi_rag: bool):
    # model, tokenizer = get_model_and_tokenizer(MODEL_NAME, for_dpo=True)

    # model = model.to(DEVICE)

    # if CHECKPOINT_PATH is not None:
    #     state_dict = torch.load(CHECKPOINT_PATH)["model_state_dict"]

    #     new_state_dict = OrderedDict()
    #     for k, v in state_dict.items():
    #         name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
    #         new_state_dict[name] = v
    #     model.load_state_dict(new_state_dict)
    # else:
    #     print("WARNING: NO CHECKPOINT PATH PROVIDED")

    # save model adapter
    ADAPTER_PATH = "/".join(CHECKPOINT_PATH.split("/")[:-1]) + "_adapter"
    # model.save_pretrained(ADAPTER_PATH)
    # return

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH, is_trainable=True)

    # model_ref = create_reference_model(model)
    model.load_adapter(ADAPTER_PATH, "reference")

    # model.eval()

    # with torch.no_grad():
    #     fake_input = "Possibly relevant resource types:\nclass github(TypedDict):\n  token: str\ngenerate the code for the following description in python: list the commits of my github repository"
    #     inputs = tokenizer(
    #         fake_input,
    #         return_tensors="pt",
    #     )
    #     inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    #     result = model.generate(**inputs, max_length=1216)
    #     print(tokenizer.decode(result[0]))

    # return

    with open("./data/processed/wm_pairs_scored.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    # random.seed(42)
    # random.shuffle(data)

    openapi_scripts = None
    if enable_openapi_rag:
        with open("data/openapi/scripts.yaml", "r") as f:
            openapi_scripts = yaml.load(f, Loader=yaml.FullLoader)
        with open("data/openapi/scripts_py.yaml", "r") as f:
            openapi_scripts += yaml.load(f, Loader=yaml.FullLoader)

    # input_sequences = [
    #     prepare_prompt(
    #         sample,
    #         model_name=MODEL_NAME,
    #         openapi_scripts=openapi_scripts,
    #     )
    #     for sample in data
    # ]
    # good = [sample["good"] for sample in data]
    # bad = [sample["bad"] for sample in data]

    # train_length = int(len(data) * 0.8)
    # train_dataset = DPODataset(
    #     input_sequences[:train_length],
    #     good[:train_length],
    #     bad[:train_length],
    # )
    # eval_dataset = DPODataset(
    #     input_sequences[train_length:],
    #     good[train_length:],
    #     bad[train_length:],
    # )

    dataset = Dataset.from_dict(
        {
            "prompt": [
                prepare_prompt(
                    sample,
                    model_name=MODEL_NAME,
                    openapi_scripts=openapi_scripts,
                )
                for sample in data
            ],
            "chosen": [sample["good"] for sample in data],
            "rejected": [sample["bad"] for sample in data],
        }
    )
    train_dataset = dataset
    # dataset = dataset.train_test_split(test_size=0.2, seed=42)
    # train_dataset = dataset["train"]
    # eval_dataset = dataset["test"]

    print("Length of train dataset", len(train_dataset))

    # input_sequences = [
    #     prepare_prompt(sample, model_name=MODEL_NAME, openapi_scripts=openapi_scripts)
    #     for sample in data
    # ]
    # good = [sample["good"] for sample in data]
    # bad = [sample["bad"] for sample in data]
    # metadata = data

    # train_set_size = int(len(data) * 0.8)
    # train_dataset = {
    #     "prompt": input_sequences[:train_set_size],
    #     "chosen": good[:train_set_size],
    #     "rejected": bad[:train_set_size],
    # }
    # train_dataset_hf =
    # eval_dataset = {
    #     "prompt": input_sequences[train_set_size:],
    #     "chosen": good[train_set_size:],
    #     "rejected": bad[train_set_size:],
    # }

    output_dir = "/".join(CHECKPOINT_PATH.split("/")[:-1]) + "_dpo"

    print("Output dir:", output_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        learning_rate=LR,
        logging_first_step=True,
        logging_steps=10,  # match results in blog post
        # evaluation_strategy="steps",
        # eval_steps=10,
        save_strategy="epoch",
        output_dir=output_dir,
        optim="rmsprop",
        warmup_steps=50,
        report_to="wandb",
        gradient_checkpointing=True,
        num_train_epochs=EPOCHS,
    )

    dpo_trainer = DPOTrainer(
        model,
        model_adapter_name="default",
        ref_adapter_name="reference",
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # data_collator=partial(collate_fn, tokenizer=tokenizer),
        # is_encoder_decoder=True,
        generate_during_eval=True,
        max_length=1216,
        max_prompt_length=512,
    )

    print("Batch size: ", dpo_trainer._train_batch_size)

    print(dpo_trainer.train)

    dpo_trainer.train()

    # model.save_pretrained(CHECKPOINT_PATH.replace(".pt", "_ppo"))


if __name__ == "__main__":
    run_dpo(ENABLE_OPENAPI_RAG)
