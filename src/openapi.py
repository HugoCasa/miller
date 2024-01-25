import requests
import yaml
import json
import os
import random
import subprocess
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

from .utils import get_resource_type_def, yaml_dump, operationId_to_summary
from .prompts import format_info


def search_openapi_schema(operation: str, app_schema: dict):
    for path, path_spec in app_schema["paths"].items():
        extra_params = {}
        for param, param_val in path_spec.items():
            if param not in ["get", "post", "put", "delete", "patch"]:
                extra_params[param] = param_val

        for method, method_spec in path_spec.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                if (
                    "summary" in method_spec and method_spec["summary"] == operation
                ) or (
                    "operationId" in method_spec
                    and operationId_to_summary(method_spec["operationId"]) == operation
                ):
                    return {
                        "openapi": app_schema["openapi"],
                        "info": {
                            "title": app_schema["info"]["title"],
                        },
                        "servers": app_schema["servers"],
                        "security": app_schema["security"]
                        if "security" in app_schema
                        else None,
                        "paths": {path: {method: {**extra_params, **method_spec}}},
                    }

    return None


def retrieve_openapi_spec(path: str):
    if path.startswith("https://"):
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.load(requests.get(path).text, Loader=yaml.FullLoader)
        elif path.endswith(".json"):
            return json.loads(requests.get(path).text)
    else:
        with open("data/openapi/" + path) as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                return yaml.load(f, Loader=yaml.FullLoader)
            elif path.endswith(".json"):
                return json.load(f)
            else:
                raise ValueError("Unsupported file format")


def save_dereferenced_openapi_schema(path: str, out: str):
    spec = retrieve_openapi_spec(path)

    if "deref" in path:
        with open(out, "w") as f:
            yaml.dump(spec, f)
        return

    with open("data/openapi/tmp.yaml", "w") as f:
        yaml.dump(spec, f)

    proc_out = subprocess.run(
        [
            "openapi",
            "bundle",
            "data/openapi/tmp.yaml",
            "--dereferenced",
            "--output",
            out,
        ],
        capture_output=True,
        text=True,
    )

    os.remove("data/openapi/tmp.yaml")

    if not os.path.isfile(out):
        print(proc_out.stderr)
        raise ValueError("Failed to dereference OpenAPI schema")


def collect_schemas():
    with open("data/openapi/index.yaml", "r") as f:
        index = yaml.load(f, Loader=yaml.FullLoader)

    for el in index:
        print(f"Retrieving {el['app']}...")
        save_dereferenced_openapi_schema(
            el["schema_url"], "data/openapi/schemas/" + el["app"] + ".yaml"
        )


MAX_HUB_SCRIPTS_PER_INTEGRATION = 40


def collect_openapi_scripts():
    with open("data/openapi/index.yaml", "r") as f:
        index = yaml.load(f, Loader=yaml.FullLoader)
    integrations = os.listdir("../windmill-integrations/hub")

    openapi_scripts = []
    for el in index:
        if el["app"] in integrations:
            actions = os.listdir(
                f"../windmill-integrations/hub/{el['app']}/scripts/action"
            )
            el["scripts"] = []
            random.seed(42)
            random.shuffle(actions)
            i = 0
            with open("data/openapi/schemas/" + el["app"] + ".yaml", "r") as f:
                app_schema = yaml.load(f, Loader=yaml.FullLoader)
            print(f"Collecting {el['app']}...")
            for action in tqdm(actions):
                script_path = f"../windmill-integrations/hub/{el['app']}/scripts/action/{action}/script.native.ts"
                if os.path.isfile(script_path):
                    if i >= MAX_HUB_SCRIPTS_PER_INTEGRATION:
                        break
                    with open(script_path) as f:
                        with open(
                            f"../windmill-integrations/hub/{el['app']}/scripts/action/{action}/script.json"
                        ) as f2:
                            code = f.read()
                            if len(code) > 1000:
                                print(
                                    f"{el['app']}'s action {action} to long, skipping..."
                                )
                                continue
                            info = json.load(f2)
                            resource_type, resource_type_def = get_resource_type_def(
                                None, code, "deno"
                            )
                            openapi_info = search_openapi_schema(
                                info["summary"], app_schema
                            )
                            if openapi_info is None:
                                print(
                                    f"Failed to find {el['app']}'s action {action} in OpenAPI schema, skipping..."
                                )
                                continue
                            openapi_scripts.append(
                                {
                                    "id": f"openapi_{el['app']}_{action}",
                                    "app": el["app"],
                                    "code": code,
                                    "lang": "deno",
                                    "instructions": info["summary"],
                                    "resource_type": resource_type,
                                    "resource_type_def": resource_type_def,
                                    "openapi_info": openapi_info,
                                }
                            )
                            i += 1
                else:
                    print(f"Skipping {el['app']}'s action {action}...")
            print("Collected scripts:", i)
        else:
            print(f"Skipping {el['app']}...")

    print(f"Saving {len(openapi_scripts)} OpenAPI scripts...")
    with open("data/openapi/scripts.yaml", "w") as f:
        yaml.dump(openapi_scripts, f)


def add_openapi_to_benchmark():
    with open("data/processed/custom_benchmark.yaml", "r") as f:
        custom_benchmark = yaml.load(f, Loader=yaml.FullLoader)

    for sample in custom_benchmark:
        if "app" in sample:
            with open("data/openapi/schemas/" + sample["app"] + ".yaml", "r") as f:
                app_schema = yaml.load(f, Loader=yaml.FullLoader)
            openapi_info = search_openapi_schema(sample["openapi_summary"], app_schema)
            if openapi_info is None:
                print(
                    f"Failed to find {sample['app']}'s action {sample['openapi_summary']} in OpenAPI schema, skipping..."
                )
            else:
                sample["openapi_info"] = openapi_info
        else:
            print(f"Skipping {sample['id']}...")

    with open("data/processed/custom_benchmark.yaml", "w") as f:
        yaml.dump(custom_benchmark, f)


def add_app_to_instructions():
    print("Adding app to instructions...")
    # with open("data/openapi/scripts.yaml", "r") as f:
    #     openapi_scripts = yaml.load(f, Loader=yaml.FullLoader)

    # for sample in openapi_scripts:
    #     sample["instructions"] = sample["instructions"] + " in " + sample["app"]

    # with open("data/openapi/scripts.yaml", "w") as f:
    #     yaml.dump(openapi_scripts, f)

    with open("data/openapi/scripts_py.yaml", "r") as f:
        openapi_scripts_py = yaml.load(f, Loader=yaml.FullLoader)

    for sample in openapi_scripts_py:
        sample["instructions"] = sample["instructions"] + " in " + sample["app"]

    with open("data/openapi/scripts_py.yaml", "w") as f:
        yaml.dump(openapi_scripts_py, f)


def create_embeddings():
    from .resource_types import compute_embedding

    schemas = os.listdir("data/openapi/schemas")

    for schema_path in tqdm(schemas):
        with open(f"data/openapi/schemas/{schema_path}", "r") as f:
            schema = yaml.load(f, Loader=yaml.FullLoader)
        embeddings = []
        for path, path_spec in schema["paths"].items():
            for method, method_spec in path_spec.items():
                if method in ["get", "post", "put", "delete", "patch"]:
                    text = ""
                    if "summary" in method_spec:
                        text += method_spec["summary"]
                    elif "description" in method_spec:
                        text += method_spec["description"]
                    elif "operationId" in method_spec:
                        text += operationId_to_summary(method_spec["operationId"])
                    else:
                        print("No summary or description for", path, method)
                    embedding = compute_embedding(text)

                    if "summary" in method_spec:
                        operation = method_spec["summary"]
                    elif "operationId" in method_spec:
                        operation = operationId_to_summary(method_spec["operationId"])
                    else:
                        print("No summary or operationId for", path, method)
                        continue

                    embeddings.append(
                        {"operation": operation, "embedding": embedding.tolist()}
                    )

        schema_name = schema_path.split(".")[0]
        with open(f"data/openapi/embeddings/{schema_name}.json", "w") as f:
            json.dump(embeddings, f)


def get_similar(text: str):
    from .resource_types import compute_embedding

    with open("data/utils/integrations_embeddings.json", "r") as f:
        integrations = json.load(f)

    with open("data/openapi/index.yaml", "r") as f:
        index = yaml.load(f, Loader=yaml.FullLoader)

    openapi_apps = [el["app"] for el in index]

    embedding = compute_embedding(text)
    queue = []
    for integration in tqdm(integrations):
        sim = embedding @ torch.tensor(integration["embedding"])
        queue.append((integration["integration"], sim))

    queue.sort(key=lambda x: x[1], reverse=True)
    integration = queue[0][0]

    if integration in openapi_apps:
        with open(f"data/openapi/embeddings/{integration}.json", "r") as f:
            embeddings = json.load(f)
        with open(f"data/openapi/schemas/{integration}.yaml", "r") as f:
            app_schema = yaml.load(f, Loader=yaml.FullLoader)
        queue = []
        for el in embeddings:
            sim = embedding @ torch.tensor(el["embedding"])
            queue.append((el["operation"], sim))

        queue.sort(key=lambda x: x[1], reverse=True)

        openapi_info = search_openapi_schema(queue[0][0], app_schema)

        for other in queue[1:3]:
            other_info = search_openapi_schema(other[0], app_schema)
            if other_info is not None:
                openapi_info["paths"] = {
                    **openapi_info["paths"],
                    **other_info["paths"],
                }

        result = format_info(openapi_info)

        return result

    return integration


if __name__ == "__main__":
    # collect_schemas()
    # collect_openapi_scripts()
    # add_openapi_to_benchmark()
    # add_app_to_instructions()
    # create_embeddings()
    print(get_similar("get a asana team by id"))
