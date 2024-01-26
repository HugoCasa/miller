from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
from peft import PeftModel
import torch
from torch import nn
from collections import OrderedDict
from peft import get_peft_model, LoraConfig, TaskType, set_peft_model_state_dict

import os

os.environ["CURL_CA_BUNDLE"] = ""


from src.models import get_model_and_tokenizer


def load_model(model_name: str, checkpoint_path: str, adapter_path: str):
    if model_name == "ise-uiuc/Magicoder-S-DS-6.7B":
        exp_name = checkpoint_path.split("/")[-2]
        if exp_name.endswith("_rag_dpo"):
            print("Loading DPO model")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            )
            model = PeftModel.from_pretrained(model, checkpoint_path)
        else:
            model, _ = get_model_and_tokenizer(model_name)

            state_dict = torch.load(checkpoint_path)["model_state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

            # save adapter
            model.save_pretrained(adapter_path)

            # reload adapter
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            )
            model = PeftModel.from_pretrained(model, adapter_path)

        # save merged
        merged_path = adapter_path.replace("_adapter", "_merged_fp16")

        model = model.merge_and_unload()
        model.save_pretrained(merged_path)

        # reload in 8bit
        model = AutoModelForCausalLM.from_pretrained(
            merged_path,
            torch_dtype=torch.float16,
            # load_in_8bit=True,
        )
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        )

        state_dict = torch.load(checkpoint_path)["model_state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


if __name__ == "__main__":
    # model_name = "salesforce/codet5p-220m"
    # checkpoint_path = (
    #     "models/salesforce/codet5p-220m_wmv2_local_2e-05_10_no_rag/model_10.pt"
    # )
    # checkpoint_path = (
    #     "models/salesforce/codet5p-220m_wmv2_local_2e-05_10_rag/model_10.pt"
    # )

    # model_name = "salesforce/codet5p-770m"
    # checkpoint_path = "/scratch/izar/casademo/pdm/models/salesforce/codet5p-770m_wmv2_dist_2e-05_10_no_rag/model_10.pt"
    # checkpoint_path = "/scratch/izar/casademo/pdm/models/salesforce/codet5p-770m_wmv2_dist_2e-05_10_rag/model_10.pt"

    model_name = "ise-uiuc/Magicoder-S-DS-6.7B"
    # checkpoint_path = "/scratch/izar/casademo/pdm/models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_no_rag/model_10.pt"
    checkpoint_path = "/scratch/izar/casademo/pdm/models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_rag/model_10.pt"
    # checkpoint_path = "/scratch/izar/casademo/pdm/models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_rag_dpo/checkpoint-625"

    exp_name = checkpoint_path.split("/")[-2]
    params_nb = model_name.split("-")[-1]
    ext = (
        "-openapi-aligned"
        if exp_name.endswith("_rag_dpo")
        else ("" if exp_name.endswith("_no_rag") else "-openapi")
    )
    hf_name = f"hugocasa/miller-{params_nb}{ext}"
    save_path = "/scratch/izar/casademo/pdm/hf_models/{}".format(hf_name)
    adapter_path = "/scratch/izar/casademo/pdm/hf_models/{}_adapter".format(hf_name)

    model, tokenizer = load_model(model_name, checkpoint_path, adapter_path)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.save_pretrained(save_path, push_to_hub=True, repo_id=hf_name)
    model.config._name_or_path = hf_name
    model.save_pretrained(
        save_path,
        push_to_hub=True,
        repo_id=hf_name,
        safe_serialization=False
        # safe_serialization=model_name == "ise-uiuc/Magicoder-S-DS-6.7B",
    )
