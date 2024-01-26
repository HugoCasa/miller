import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration
import torch
import json

from src.prompts import (
    SIMPLE_PROMPT,
    SIMPLE_EDIT_PROMPT,
    SIMPLE_FIX_PROMPT,
    MAGICODER_PROMPT,
    MAGICODER_FIX_PROMPT,
    MAGICODER_EDIT_PROMPT,
    OPENAPI_INFO_PROMPT,
    resource_types,
)
from src.resource_types import (
    compute_embedding,
    get_similar_resource_types,
)
from src.openapi import get_similar_openapi


def load_rts_and_embeddings():
    with open("./data/utils/integrations_embeddings.json", "r") as f:
        resource_types_embeddings = json.load(f)
    return resource_types, resource_types_embeddings


def generate(
    lang: str,
    instructions=None,
    code=None,
    error=None,
    kind="gen",
    model_name="hugocasa/miller-6.7B-openapi-aligned",
    device="cuda",
    verbose=False,
):
    rts, rts_embeddings = load_rts_and_embeddings()
    if kind == "gen":
        prompt = MAGICODER_PROMPT if "6.7B" in model_name else SIMPLE_PROMPT
        emb = compute_embedding(instructions)
        resource_types = "\n".join(
            get_similar_resource_types(emb, lang, rts, rts_embeddings)
        )
        if "openapi" in model_name:
            openapi_info = get_similar_openapi(emb)
            if openapi_info:
                openapi_info_prompt = OPENAPI_INFO_PROMPT.format(
                    openapi_info=openapi_info
                )
                prompt = prompt.format(
                    instructions=instructions,
                    resource_types=resource_types,
                    lang=lang,
                    openapi_info_prompt=openapi_info_prompt,
                )
            else:
                prompt = prompt.format(
                    instructions=instructions,
                    resource_types=resource_types,
                    lang=lang,
                    openapi_info_prompt="",
                )
        else:
            prompt = prompt.format(
                instructions=instructions,
                resource_types=resource_types,
                lang=lang,
                openapi_info_prompt="",
            )
    elif kind == "fix":
        prompt = MAGICODER_FIX_PROMPT if "6.7B" in model_name else SIMPLE_FIX_PROMPT
        prompt = prompt.format(code=code, error=error, lang=lang)
    elif kind == "edit":
        prompt = MAGICODER_EDIT_PROMPT if "6.7B" in model_name else SIMPLE_EDIT_PROMPT
        prompt = prompt.format(instructions=instructions, code=code, lang=lang)
    else:
        raise ValueError("Invalid kind")

    if verbose:
        print(prompt)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "6.7B" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
    model = model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str)
    parser.add_argument("--instructions", type=str, default=None)
    parser.add_argument("--code", type=str, default=None)
    parser.add_argument("--error", type=str, default=None)
    parser.add_argument("--kind", type=str, default="gen")
    parser.add_argument("--model_name", type=str, default="hugocasa/miller-6.7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", type=str, default=False)
    args = parser.parse_args()

    print(
        generate(
            args.lang,
            instructions=args.instructions,
            code=args.code,
            error=args.error,
            kind=args.kind,
            model_name=args.model_name,
            device=args.device,
            verbose=args.verbose,
        )
    )

    # instructions = "return the number of commits of a given github repository"
    # kind = "gen"
    # model_name = "hugocasa/miller-6.7B-openapi-aligned"
    # # model_name = "hugocasa/miller-220m-openapi"
    # lang = "python"
    # print(
    #     generate(
    #         lang,
    #         instructions=instructions,
    #         kind=kind,
    #         model_name=model_name,
    #         device="cuda",
    #         verbose=True,
    #     )
    # )

    # code = "def main(nb: int): \nreturn nb2"
    # error = "NameError: name 'nb2' is not defined"
    # kind = "fix"
    # # model_name = "hugocasa/miller-6.7B-openapi-aligned"
    # model_name = "hugocasa/miller-220m-openapi"
    # lang = "python"
    # print(
    #     generate(
    #         lang,
    #         code=code,
    #         error=error,
    #         kind=kind,
    #         model_name=model_name,
    #         device="cuda",
    #         verbose=True,
    #     )
    # )
