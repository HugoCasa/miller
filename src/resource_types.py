from transformers import AutoTokenizer, AutoModel
import json
import torch
from torch import Tensor
from torch.nn import functional as F
from .utils import python_compile, ts_compile, to_camel_case, average_pool
from tqdm import tqdm
import yaml


tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
model = AutoModel.from_pretrained("thenlper/gte-small")


def compute_embedding(text: str):
    input = tokenizer(text, max_length=128, truncation=True, return_tensors="pt")
    outputs = model(**input)

    embeddings = average_pool(outputs.last_hidden_state, input.attention_mask)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings[0]


def format_resource_type(rt: dict, lang: str):
    if lang == "deno" or lang == "bun":
        return f"type {to_camel_case(rt['name']).title()} = {ts_compile(json.loads(rt['schema']))}"
    elif lang == "python":
        return f"class {rt['name']}(TypedDict):\n{python_compile(json.loads(rt['schema']))}"
    else:
        raise Exception(f"Unknown language {lang}")


def get_similar_resource_types(
    embedding: str, lang: str, resource_types, resource_types_embeddings, limit: int = 3
):
    queue = []
    for rt_emb in resource_types_embeddings:
        sim = embedding @ torch.tensor(rt_emb["embedding"])
        queue.append((rt_emb["integration"], sim))

    queue.sort(key=lambda x: x[1], reverse=True)

    return list(
        map(
            lambda x: [
                y["resource_type_def"]
                for y in resource_types
                if y["lang"] == lang
                and (
                    y["resource_type"] == x[0]
                    if lang == "python"
                    else y["resource_type"] == to_camel_case(x[0]).title()
                )
            ][0],
            queue[:limit],
        )
    )


def compute_integration_embeddings():
    with open("./data/utils/integrations.yaml", "r") as f:
        actions = yaml.load(f, Loader=yaml.FullLoader)

    # group actions by the integration key
    actions_by_integration = {}
    for action in actions:
        if action["integration"] not in actions_by_integration:
            actions_by_integration[action["integration"]] = []
        actions_by_integration[action["integration"]].append(action)

    # compute the embedding for each integration
    integrations = []
    for integration, actions in tqdm(actions_by_integration.items()):
        actions_embeddings = []
        for action in actions:
            action_str = f"{integration};{action['name']};{action['description']}"
            actions_embeddings.append(compute_embedding(action_str))
        actions_embeddings = torch.stack(actions_embeddings)
        integration_embedding = torch.mean(actions_embeddings, dim=0).tolist()
        integrations.append(
            {
                "integration": integration,
                "embedding": integration_embedding,
            }
        )

    with open("./data/utils/integrations_embeddings.json", "w") as f:
        json.dump(integrations, f)


if __name__ == "__main__":
    compute_integration_embeddings()
    # print(get_similar_resource_types("send message", "python"))
    # print(get_similar_resource_types("get email"))
