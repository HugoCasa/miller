from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    TextStreamer,
)
import torch
import time
import numpy as np

DEVICE = torch.device("cuda")


def time_hf_model(input_text: str, path: str):
    if "220m" in path or "770m" in path:
        model = T5ForConditionalGeneration.from_pretrained(
            path, torch_dtype=torch.float16
        )
        model = model.to(DEVICE)
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16)
        model = model.to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(path.replace("_merged_fp16", ""))

    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)

    speeds = []
    for _ in range(5):
        start_time = time.time()
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,
        )
        end_time = time.time()
        length = len(outputs[0])
        if "6.7B" in path:
            length = length - len(inputs["input_ids"][0])
        speeds.append(length / (end_time - start_time))
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print(f"{path}: {np.mean(speeds)} tokens/s ({np.std(speeds)})")


if __name__ == "__main__":
    text_codet5p = """
    Possibly relevant resource types:
    class cloudflare_api_key(TypedDict):
        email: str
        api_key: str
    class asana(TypedDict):
        token: str
    class nethunt_crm(TypedDict):
        api_key: str
        base_url: str

    generate the code for the following description in deno: Get users from a team in Asana
    """

    text_magicoder = """
    You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

    @@ Instruction
    Possibly relevant resource types:
    class cloudflare_api_key(TypedDict):
        email: str
        api_key: str
    class asana(TypedDict):
        token: str
    class nethunt_crm(TypedDict):
        api_key: str
        base_url: str

    generate the code for the following description in python: return the number of users in a given asana team

    @@ Response
    """
    time_hf_model(
        text_codet5p, "/scratch/izar/casademo/pdm/hf_models/hugocasa/miller-220m"
    )
    time_hf_model(
        text_codet5p, "/scratch/izar/casademo/pdm/hf_models/hugocasa/miller-770m"
    )
    time_hf_model(
        text_magicoder,
        "/scratch/izar/casademo/pdm/hf_models/hugocasa/miller-6.7B_merged_fp16",
    )
