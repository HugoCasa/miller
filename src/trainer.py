import os
import logging
from typing import Dict

import submitit
import wandb

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import json
from tqdm import tqdm
from functools import partial

from .datasets import collate_fn_cond, collate_fn_causal
from .models import get_model_and_tokenizer

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# torch.autograd.set_detect_anomaly(True)


def get_collate_fn(tokenizer, model_name):
    if (
        model_name == "salesforce/codet5p-220m"
        or model_name == "salesforce/codet5p-770m"
    ):
        return partial(collate_fn_cond, tokenizer=tokenizer)
    elif model_name == "ise-uiuc/Magicoder-S-DS-6.7B":
        return partial(collate_fn_causal, tokenizer=tokenizer)
    else:
        raise ValueError(f"Model {model_name} not supported")


class Trainer:
    def __init__(
        self,
        lr: float,
        batch_size: int,
        val_batch_size: int,
        epochs: int,
        save_dir: str,
        model_name: str,
        train_dataset: Dataset = None,
        val_dataset: Dataset = None,
        wandb_config: Dict = None,
        checkpoint_path: str = None,
        local=False,
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.epochs = epochs
        self.save_dir = save_dir
        self.checkpoint_path = checkpoint_path
        self.local = local
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.wandb_config = wandb_config

    def __call__(self):
        if self.local:
            self._setup_local()
        else:
            self._setup_slurm()

        if self.dist_env.rank == 0:
            config = {
                "lr": self.lr,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "val_batch_size": self.val_batch_size,
            }
            wandb.init(**self.wandb_config, config=config)
        self._train()

    def _setup_slurm(self):
        self.dist_env = (
            submitit.helpers.TorchDistributedEnvironment().export()
        )  # export the variables for distributed training

        LOG.info(
            f"Process group: {self.dist_env.world_size} tasks, rank: {self.dist_env.rank}"
        )
        dist.init_process_group("nccl")

    def _setup_local(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=0, world_size=1)

        class dotdict(dict):
            """dot.notation access to dictionary attributes"""

            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        self.dist_env = dotdict(
            {
                "world_size": 1,
                "rank": 0,
            }
        )

    def _get_dataloader(self):
        """Get training and validation dataloaders"""

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # done by the sampler
            pin_memory=True,
            num_workers=0,
            sampler=DistributedSampler(self.train_dataset),
            collate_fn=self.collate_fn,
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,  # done by the sampler
            pin_memory=True,
            num_workers=0,
            sampler=DistributedSampler(self.val_dataset),
            collate_fn=self.collate_fn,
        )

        return train_loader, val_loader

    def setup_model_and_tokenizer(self):
        self.model, self.tokenizer = get_model_and_tokenizer(self.model_name)
        self.collate_fn = get_collate_fn(self.tokenizer, self.model_name)

    def _train(self):
        local_rank = 0  # as one task per gpu, device is always 0

        self.setup_model_and_tokenizer()

        # wrap model for distributed training
        model = self.model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])

        # if self.dist_env.rank == 0:
        #     wandb.watch(model, log_freq=10)

        train_loader, val_loader = self._get_dataloader()

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        if self.checkpoint_path is not None:
            # if checkpoint path given, load weights
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            last_epoch = checkpoint["epoch"]
            LOG.info(f"Loaded checkpoint ({last_epoch+1} epochs)")
        else:
            last_epoch = -1

            LOG.info("Validation base model\n-------------------------------")

            loss = self.validation(val_loader, local_rank, model)

            # with open(
            #     f"{self.save_dir}/outputs_baseline_{self.dist_env.rank}.json", "w"
            # ) as f:
            #     f.write(json.dumps(outputs, indent=2))

        dist.barrier()
        LOG.info("Initialization passed successfully.")

        for epoch in range(self.epochs):
            if epoch <= last_epoch:
                LOG.info(f"Epoch {epoch+1} already trained")
                continue

            LOG.info(f"Epoch {epoch+1}\n-------------------------------")

            model.train()

            train_loader.sampler.set_epoch(epoch)

            loss_sum = 0
            for batch_idx, data in enumerate(train_loader):
                input_ids = data["input_ids"].to(local_rank)
                attention_mask = data["attention_mask"].to(local_rank)
                labels = data["labels"].to(local_rank)
                loss = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                ).loss

                loss_sum += loss.item()

                # backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # log loss
                current = self.batch_size * (batch_idx + 1) * self.dist_env.world_size
                total = len(self.train_dataset)
                print(
                    f"\nloss: {loss.item():>7f}  [{current:>5d}/{total:>5d}]",
                    flush=True,
                )

                if self.dist_env.rank == 0:
                    wandb.log(
                        {
                            "loss": loss.item(),
                        }
                    )

            if self.dist_env.rank == 0:
                wandb.log({"mean_loss": loss_sum / (batch_idx + 1)})

            LOG.info("Validation\n-------------------------------")
            loss = self.validation(val_loader, local_rank, model)

            # LOG.info("Saving generations\n-------------------------------")
            # with open(
            #     f"{self.save_dir}/outputs_{epoch+1}_{self.dist_env.rank}.json", "w"
            # ) as f:
            #     f.write(json.dumps(outputs, indent=2))

            if self.dist_env.rank == 0:  # global rank = 0
                LOG.info("Saving model\n-------------------------------")
                # save model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    },
                    f"{self.save_dir}/model_{epoch+1}.pt",
                )

    def validation(
        self, val_loader: DataLoader, local_rank: int, model: torch.nn.Module
    ):
        model.eval()

        with torch.no_grad():
            loss_sum = 0
            for batch_idx, data in enumerate(val_loader):
                input_ids = data["input_ids"].to(local_rank)
                attention_mask = data["attention_mask"].to(local_rank)
                labels = data["labels"].to(local_rank)
                # metadata = data["metadata"]
                loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                ).loss

                loss_sum += loss.item()

                # log loss
                current = (
                    self.val_batch_size * (batch_idx + 1) * self.dist_env.world_size
                )
                total = len(self.val_dataset)
                print(
                    f"\nloss: {loss.item():>7f}  [{current:>5d}/{total:>5d}]",
                    flush=True,
                )

        if self.dist_env.rank == 0:
            wandb.log({"eval_mean_loss": loss_sum / (batch_idx + 1)})

        return loss

    def inference(
        self,
        input_texts: list[str],
        batch_size=4,
    ):
        if self.local:
            self._setup_local()
        else:
            raise NotImplementedError
            # self._setup_slurm()

        local_rank = 0
        self.setup_model_and_tokenizer()
        model = self.model.to(local_rank)
        model = DDP(model, device_ids=[local_rank])

        if self.checkpoint_path is not None:
            print("Loading checkpoint at ", self.checkpoint_path)
            checkpoint = torch.load(self.checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        with torch.no_grad():
            outputs = []
            indices = list(range(0, len(input_texts), batch_size))
            for i in tqdm(indices):
                batch_texts = input_texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=512,
                )
                batch_outputs = model.module.generate(
                    input_ids=inputs["input_ids"].to(local_rank),
                    attention_mask=inputs["attention_mask"].to(local_rank),
                    max_new_tokens=512,
                )
                outputs.extend(
                    [
                        self.tokenizer.decode(
                            batch_outputs[idx], skip_special_tokens=True
                        )
                        for idx in range(batch_outputs.shape[0])
                    ]
                )
                break

            return outputs
