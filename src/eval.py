import json
from transformers import AutoTokenizer, AutoModel
import os
import torch
from torch.nn import functional as F
from tqdm import tqdm
from typing import Union, List
import re
import yaml


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")


def setup_embedding_model():
    model = AutoModel.from_pretrained(
        "salesforce/codet5p-110m-embedding", trust_remote_code=True
    )
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "salesforce/codet5p-110m-embedding", trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def compute_similarity(model, tokenizer, a: str, b: str):
    inputs = tokenizer(
        [f"{sentence}" for sentence in [a, b]],
        padding="longest",
        max_length=1024,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embeddings = model(**inputs)
        a_embedding = embeddings[0]
        b_embedding = embeddings[1]
        return a_embedding @ b_embedding


def check_syntax(code, lang="deno"):
    if lang == "python":
        with open(".tmp.py", "w") as f:
            f.write(code)
        result = os.popen("ruff check .tmp.py").read()
        os.remove(".tmp.py")
        if result.find("Found") != -1:
            return 0
        else:
            return 1
    elif lang == "deno":
        with open(".tmp.ts", "w") as f:
            f.write(code)
        result = os.popen("deno lint --json .tmp.ts").read()
        os.remove(".tmp.ts")
        # parse json result
        result = json.loads(result)
        if len(result["errors"]) > 0:
            return 0
        else:
            return 1
    else:
        return 1


def match_resource_types(code: str, lang: str):
    if lang == "python":
        matches = re.findall(r"class (\S+)\(TypedDict\)", code)
        return matches
    elif lang == "deno":
        matches = re.findall(r"type (\S+) = {", code)
        return matches
    else:
        print(f"Unknown language {lang}")
        return []


def check_resource_type(
    code: str,
    lang: str,
    resource_type: str,
):
    matches = match_resource_types(code, lang)

    if resource_type in matches:
        return 1
    else:
        return 0


def check_main_func(code: str, lang: str):
    if lang == "python":
        match = re.search("^def main\([^\)]*\)", code, re.MULTILINE)
        return 1 if match else 0
    elif lang == "deno":
        match = re.search("^export async function main\([^\)]*\)", code, re.MULTILINE)
        return 1 if match else 0
    return 0


def evaluate_results(path: str):
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    eval_out = path.replace("_samples.yaml", "_results.yaml")
    stats_out = path.replace("_samples.yaml", "_stats.yaml")

    model, tokenizer = setup_embedding_model()
    results = []
    for sample in tqdm(data):
        scores = {}

        sample = {
            **sample,
            **sample["metadata"],
            "response": re.sub(
                r".*@@ Response\n", "", sample["response"], flags=re.DOTALL
            ),
            "metadata": None,
        }

        kind = "gen"
        if sample["id"].startswith("hubfix_"):
            kind = "fix"
        elif sample["id"].startswith("hubedit_"):
            kind = "edit"

        # embedding compute_similarity between generated code and the original code
        if kind == "gen":
            scores["similarity"] = (
                compute_similarity(model, tokenizer, sample["code"], sample["response"])
            ).item()
        elif kind == "edit":
            scores["similarity"] = (
                compute_similarity(
                    model, tokenizer, sample["modified_code"], sample["response"]
                )
            ).item()
        elif kind == "fix":
            scores["similarity"] = (
                compute_similarity(
                    model, tokenizer, sample["original_code"], sample["response"]
                )
            ).item()
        else:
            raise Exception(f"Unknown kind {kind}")

        # syntax checking of generated code
        scores["syntax"] = check_syntax(sample["response"], sample["lang"])

        # main func check
        scores["main_func"] = check_main_func(sample["response"], sample["lang"])

        if "resource_type" in sample and sample["resource_type"] is not None:
            scores["resource_type"] = check_resource_type(
                sample["response"], sample["lang"], sample["resource_type"]
            )
        else:
            # there should be no resource type in the generated code
            rts = match_resource_types(sample["response"], sample["lang"])
            if len(rts) > 0:
                scores["resource_type"] = 0
            else:
                scores["resource_type"] = 1

        results.append(
            {
                **sample,
                "scores": scores,
                "score": sum(scores.values()) / len(scores.values()),
            }
        )

    with open(eval_out, "w") as f:
        yaml.dump(results, f)

    gen_results = [
        r
        for r in results
        if not r["id"].startswith("hubfix_") and not r["id"].startswith("hubedit_")
    ]
    fix_results = [r for r in results if r["id"].startswith("hubfix_")]
    edit_results = [r for r in results if r["id"].startswith("hubedit_")]

    stats = {
        "similarity": {
            "overall": sum(map(lambda x: x["scores"]["similarity"], results))
            / len(results),
            "gen": sum(map(lambda x: x["scores"]["similarity"], gen_results))
            / len(gen_results),
            "fix": sum(map(lambda x: x["scores"]["similarity"], fix_results))
            / len(fix_results),
            "edit": sum(map(lambda x: x["scores"]["similarity"], edit_results))
            / len(edit_results),
        },
        "syntax": {
            "overall": sum(map(lambda x: x["scores"]["syntax"], results))
            / len(results),
            "gen": sum(map(lambda x: x["scores"]["syntax"], gen_results))
            / len(gen_results),
            "fix": sum(map(lambda x: x["scores"]["syntax"], fix_results))
            / len(fix_results),
            "edit": sum(map(lambda x: x["scores"]["syntax"], edit_results))
            / len(edit_results),
        },
        "main_func": {
            "overall": sum(map(lambda x: x["scores"]["main_func"], results))
            / len(results),
            "gen": sum(map(lambda x: x["scores"]["main_func"], gen_results))
            / len(gen_results),
            "fix": sum(map(lambda x: x["scores"]["main_func"], fix_results))
            / len(fix_results),
            "edit": sum(map(lambda x: x["scores"]["main_func"], edit_results))
            / len(edit_results),
        },
        "resource_type": {
            "overall": sum(map(lambda x: x["scores"]["resource_type"], results))
            / len(results),
            "gen": sum(map(lambda x: x["scores"]["resource_type"], gen_results))
            / len(gen_results),
            "fix": sum(map(lambda x: x["scores"]["resource_type"], fix_results))
            / len(fix_results),
            "edit": sum(map(lambda x: x["scores"]["resource_type"], edit_results))
            / len(edit_results),
        },
        "overall": {
            "overall": sum(map(lambda x: x["score"], results)) / len(results),
            "gen": sum(map(lambda x: x["score"], gen_results)) / len(gen_results),
            "fix": sum(map(lambda x: x["score"], fix_results)) / len(fix_results),
            "edit": sum(map(lambda x: x["score"], edit_results)) / len(edit_results),
        },
    }

    with open(stats_out, "w") as f:
        yaml.dump(stats, f)


if __name__ == "__main__":
    # import sys

    # path = sys.argv[1]

    path = (
        # "models/salesforce/codet5p-220m_wmv2_local_2e-05_10_no_rag/validation_samples.yaml"
        # "models/salesforce/codet5p-220m_wmv2_local_2e-05_10_rag/validation_samples.yaml"
        # "models/salesforce/codet5p-770m_wmv2_dist_2e-05_10_no_rag/validation_samples.yaml"
        # "models/salesforce/codet5p-770m_wmv2_dist_2e-05_10_rag/validation_samples.yaml"
        # "models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_no_rag/validation_samples.yaml"
        "models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_rag/validation_samples.yaml"
    )

    evaluate_results(path)
