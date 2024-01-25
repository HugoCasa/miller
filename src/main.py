import submitit
import logging
import datetime
import yaml
import os

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# Training parameters
LR = 2e-5
BATCH_SIZE = 4  # batch size per GPU
VAL_BATCH_SIZE = 4
EPOCHS = 10

ENABLE_OPENAPI_RAG = True


LOCAL = False

# MODEL_NAME = "salesforce/codet5p-770m"
MODEL_NAME = "ise-uiuc/Magicoder-S-DS-6.7B"
EXP_NAME = f"{MODEL_NAME}_wmv2_{'local' if LOCAL else 'dist'}_{LR}_{EPOCHS}_{'rag' if ENABLE_OPENAPI_RAG else 'no_rag'}"

# Basic parameters
# SAVE_DIR = f"/home/casademo/pdm/models/{EXP_NAME}"
SAVE_DIR = f"/scratch/izar/casademo/pdm/models/{EXP_NAME}"
# CHECKPOINT_PATH = None  # f"/home/casademo/pdm/models/{EXP_NAME}/model_9.pt"
CHECKPOINT_PATH = None  # f"/scratch/izar/casademo/pdm/models/{EXP_NAME}/model_6.pt"


# SLURM parameters
LOG_DIR = f"/home/casademo/pdm/logs/{EXP_NAME}"
N_NODES = 2
GPUS_PER_NODE = 2
CPUS_PER_NODE = 40
MEM_PER_NODE = 150


from .trainer import Trainer
from .datasets import get_datasets


def main(model_name=MODEL_NAME):
    train_dataset, val_dataset = get_datasets(
        model_name, enable_openapi_rag=ENABLE_OPENAPI_RAG
    )

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    trainer = Trainer(
        lr=LR,
        batch_size=BATCH_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        epochs=EPOCHS,
        save_dir=SAVE_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        local=LOCAL,
        model_name=model_name,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        wandb_config={
            "project": "pdm",
            "name": EXP_NAME
            + "_"
            + datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "group": EXP_NAME,
        },
    )

    if LOCAL:
        LOG.info("Running locally")
        trainer()
        return 0

    executor = submitit.AutoExecutor(folder=LOG_DIR)
    executor.update_parameters(
        name="train_code_llm",
        nodes=N_NODES,
        mem_gb=MEM_PER_NODE,
        gpus_per_node=GPUS_PER_NODE,
        tasks_per_node=GPUS_PER_NODE,
        cpus_per_task=CPUS_PER_NODE // GPUS_PER_NODE,  # 40 total on one node
        timeout_min=60 * 20,
        slurm_partition="gpu",
        slurm_qos="gpu",
        slurm_gres=f"gpu:{GPUS_PER_NODE}",
        slurm_additional_parameters={
            "requeue": True,
            "account": "master",
        },
    )

    job = executor.submit(trainer)

    LOG.info(f"Submitted job_id: {job.job_id}")

    return job


if __name__ == "__main__":
    main()
