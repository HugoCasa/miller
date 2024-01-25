import yaml
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
from tqdm import tqdm

from .prompts import prepare_prompt
from .trainer import Trainer
from .datasets import get_datasets

# MODEL_NAME = "salesforce/codet5p-220m"
# MODEL_NAME = "salesforce/codet5p-770m"
MODEL_NAME = "ise-uiuc/Magicoder-S-DS-6.7B"

ADAPTER_PATH = "/scratch/izar/casademo/pdm/models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_rag_dpo/checkpoint-625"
MERGED_PATH = "/scratch/izar/casademo/pdm/models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_rag_dpo/checkpoint-625-merged"

# CHECKPOINT_PATH = "models/salesforce/codet5p-220m_wmv2_local_2e-05_10_rag/model_10.pt"
# CHECKPOINT_PATH = (
#     "models/salesforce/codet5p-220m_wmv2_local_2e-05_10_no_rag/model_10.pt"
# )
# CHECKPOINT_PATH = "/scratch/izar/casademo/pdm/models/salesforce/codet5p-770m_wmv2_dist_2e-05_10_no_rag/model_10.pt"
# CHECKPOINT_PATH = "/scratch/izar/casademo/pdm/models/salesforce/codet5p-770m_wmv2_dist_2e-05_10_rag/model_10.pt"
# CHECKPOINT_PATH = "/scratch/izar/casademo/pdm/models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_no_rag/model_10.pt"
CHECKPOINT_PATH = "/scratch/izar/casademo/pdm/models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_rag/model_10.pt"

ENABLE_OPENAPI_RAG = True

DEVICE = torch.device("cuda:0")


def get_inference_trainer():
    trainer = Trainer(
        lr=0,
        batch_size=0,
        val_batch_size=0,
        epochs=0,
        save_dir="",
        checkpoint_path=CHECKPOINT_PATH,
        local=True,
        model_name=MODEL_NAME,
    )
    return trainer


def get_peft_model():
    # model = AutoModelForCausalLM.from_pretrained(
    #     MODEL_NAME,
    #     torch_dtype=torch.float16,
    # )
    # model = PeftModel.from_pretrained(
    #     model,
    #     ADAPTER_PATH,
    #     torch_dtype=torch.float16,
    # )
    # model = model.merge_and_unload()
    # model.save_pretrained(MERGED_PATH)

    model = AutoModelForCausalLM.from_pretrained(MERGED_PATH, load_in_8bit=True)
    return model


def inference_single(input_text: str, lang: str):
    trainer = get_inference_trainer()
    prompt = prepare_prompt(
        {"lang": lang, "instructions": input_text}, model_name=MODEL_NAME
    )
    print("Prompt:", prompt)
    result = trainer.inference([prompt])
    print("Answer:", result[0])


def inference(
    model,
    tokenizer,
    input_texts: list[str],
    batch_size=4,
):
    model.eval()
    with torch.no_grad():
        outputs = []
        indices = list(range(0, len(input_texts), batch_size))
        for i in tqdm(indices):
            batch_texts = input_texts[i : i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=512,
            )
            batch_outputs = model.generate(
                input_ids=inputs["input_ids"].to(DEVICE),
                attention_mask=inputs["attention_mask"].to(DEVICE),
                max_new_tokens=512,
            )
            outputs.extend(
                [
                    tokenizer.decode(batch_outputs[idx], skip_special_tokens=True)
                    for idx in range(batch_outputs.shape[0])
                ]
            )

        return outputs


def gen_validation_samples(out: str, enable_openapi_rag=False):
    _, val_dataset = get_datasets(MODEL_NAME, enable_openapi_rag=enable_openapi_rag)
    trainer = get_inference_trainer()

    text_inputs = [inputs["input_sequences"] for inputs in val_dataset.data]
    outputs = trainer.inference(
        text_inputs, 8 if MODEL_NAME == "salesforce/codet5p-220m" else 4
    )
    results = [
        {
            **sample,
            "response": output,
        }
        for sample, output in zip(val_dataset.data, outputs)
    ]
    with open(out, "w") as f:
        yaml.dump(results, f)


def gen_benchmark_samples(out: str, enable_openapi_rag=False):
    with open("./data/processed/benchmark_dataset_v2.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    trainer = get_inference_trainer()

    openapi_scripts = None
    if enable_openapi_rag:
        with open("data/openapi/scripts.yaml", "r") as f:
            openapi_scripts = yaml.load(f, Loader=yaml.FullLoader)
        with open("data/openapi/scripts_py.yaml", "r") as f:
            openapi_scripts += yaml.load(f, Loader=yaml.FullLoader)

    prompts = [
        prepare_prompt(d, model_name=MODEL_NAME, openapi_scripts=openapi_scripts)
        for d in data
    ]
    responses = trainer.inference(
        prompts, 8 if MODEL_NAME == "salesforce/codet5p-220m" else 4
    )

    results = []
    for d, r in zip(data, responses):
        results.append(
            {
                **d,
                "openapi_info": None,
                "response": r,
            }
        )

    with open(out, "w") as f:
        yaml.dump(results, f)


def gen_benchmark_samples_dpo(out: str, enable_openapi_rag=False):
    with open("./data/processed/benchmark_dataset_v2.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    model = get_peft_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    openapi_scripts = None
    if enable_openapi_rag:
        with open("data/openapi/scripts.yaml", "r") as f:
            openapi_scripts = yaml.load(f, Loader=yaml.FullLoader)
        with open("data/openapi/scripts_py.yaml", "r") as f:
            openapi_scripts += yaml.load(f, Loader=yaml.FullLoader)

    prompts = [
        prepare_prompt(d, model_name=MODEL_NAME, openapi_scripts=openapi_scripts)
        for d in data
    ]
    responses = inference(model, tokenizer, prompts, 4)

    results = []
    for d, r in zip(data, responses):
        results.append(
            {
                **d,
                "openapi_info": None,
                "response": r,
            }
        )

    with open(out, "w") as f:
        yaml.dump(results, f)


if __name__ == "__main__":
    # import sys

    # input_text = sys.argv[1]
    # inference_single(input_text)
    # gen_benchmark_samples(
    #     # "./models/salesforce/codet5p-220m_wmv2_local_2e-05_10_rag/benchmark_samples.yaml",
    #     # "./models/salesforce/codet5p-220m_wmv2_local_2e-05_10_no_rag/benchmark_samples.yaml",
    #     # "./models/salesforce/codet5p-770m_wmv2_dist_2e-05_10_no_rag/benchmark_samples.yaml",
    #     # "./models/salesforce/codet5p-770m_wmv2_dist_2e-05_10_rag/benchmark_samples.yaml",
    #     # "./models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_no_rag/benchmark_samples.yaml",
    #     # "./models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_rag/benchmark_samples2.yaml",
    #     ENABLE_OPENAPI_RAG,
    # )
    gen_benchmark_samples_dpo(
        "./models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_rag_dpo/benchmark_samples.yaml",
        ENABLE_OPENAPI_RAG,
    )
    # gen_validation_samples(
    #     # "./models/salesforce/codet5p-220m_wmv2_local_2e-05_10_rag/validation_samples.yaml",
    #     # "./models/salesforce/codet5p-220m_wmv2_local_2e-05_10_no_rag/validation_samples.yaml",
    #     # "./models/salesforce/codet5p-770m_wmv2_dist_2e-05_10_no_rag/validation_samples.yaml",
    #     # "./models/salesforce/codet5p-770m_wmv2_dist_2e-05_10_rag/validation_samples.yaml",
    #     # "./models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_no_rag/validation_samples.yaml",
    #     "./models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_rag/validation_samples.yaml",
    #     ENABLE_OPENAPI_RAG,
    # )
