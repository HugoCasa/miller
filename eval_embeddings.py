import yaml
import torch
import json
from tqdm import tqdm
import re

from src.resource_types import compute_embedding
from src.openapi import operationId_to_summary


def test_openapi():
    with open("data/openapi/scripts.yaml", "r") as f:
        openapi_scripts = yaml.load(f, Loader=yaml.FullLoader)
        with open("data/openapi/scripts_py.yaml", "r") as f:
            scripts_py = yaml.load(f, Loader=yaml.FullLoader)
            openapi_scripts.extend(scripts_py)

    def get_similar_openapi_summaries(embedding: str, embeddings: list[dict], limit=3):
        queue = []
        for rt_emb in embeddings:
            sim = embedding @ torch.tensor(rt_emb["embedding"])
            queue.append((rt_emb["operation"], sim))

        queue.sort(key=lambda x: x[1], reverse=True)

        return list(
            map(
                lambda x: x[0],
                queue[:limit],
            )
        )

    top1 = 0
    top3 = 0
    for sample in (pbar := tqdm(openapi_scripts)):
        embedding = compute_embedding(sample["instructions"])
        # instructions =
        app = sample["app"]

        with open(f"data/openapi/embeddings/{app}.json", "r") as f:
            embeddings = json.load(f)

        sims = get_similar_openapi_summaries(embedding, embeddings)
        openapi_summary = " in ".join(sample["instructions"].split(" in ")[:-1])

        if openapi_summary in sims:
            top3 += 1
        if openapi_summary == sims[0]:
            top1 += 1

        pbar.set_description(f"Top 1 {top1}; Top 3 {top3}")

    print(f"Top 1: {top1 / len(openapi_scripts)}")
    print(f"Top 3: {top3 / len(openapi_scripts)}")


def test_integration():
    with open("data/processed/wm_dataset_v2.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        data = [d for d in data if d["id"].startswith("synthetic_")]

    with open("data/utils/integrations_embeddings.json") as f:
        integration_embeddings = json.load(f)

    def get_similar_integrations(embedding: str, limit=3):
        queue = []
        for rt_emb in integration_embeddings:
            sim = embedding @ torch.tensor(rt_emb["embedding"])
            queue.append((rt_emb["integration"], sim))

        queue.sort(key=lambda x: x[1], reverse=True)

        return list(
            map(
                lambda x: x[0],
                queue[:limit],
            )
        )

    top1 = 0
    top3 = 0
    for sample in tqdm(data):
        # splitted = sample["instructions"].split(" ")
        # instructions = " ".join(splitted[:-2])
        instructions = " in ".join(sample["instructions"].split(" in ")[:-1])
        embedding = compute_embedding(instructions)
        sims = get_similar_integrations(embedding)

        app = sample["instructions"].split(" in ")[-1].replace(" ", "_")

        if app in sims:
            top3 += 1
        # else:
        # print("Instructions: ", instructions)
        # print(f"App: {app}")
        # print(f"Similar: {sims}")
        if app == sims[0]:
            top1 += 1

    print(f"Top 1: {top1 / len(data)}")
    print(f"Top 3: {top3 / len(data)}")


if __name__ == "__main__":
    test_openapi()
    # test_integration()
