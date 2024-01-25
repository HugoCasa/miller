from typing import TypedDict
import random
import yaml
import os
from .utils import operationId_to_summary

SIMPLE_PROMPT = """Possibly relevant resource types:
{resource_types}
{openapi_info_prompt}
generate the code for the following description in {lang}: {instructions}"""

SIMPLE_EDIT_PROMPT = """{code}
edit the code in {lang} as instructed: {instructions}"""

SIMPLE_FIX_PROMPT = """{code}
the code above in {lang} has the following error: {error}
fix the code"""

MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
Possibly relevant resource types:
{resource_types}
{openapi_info_prompt}
generate the code for the following description in {lang}: {instructions}

@@ Response
"""

MAGICODER_EDIT_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{code}
edit the code in {lang} as instructed: {instructions}

@@ Response
"""

MAGICODER_FIX_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{code}
the code above in {lang} has the following error: {error}
fix the code

@@ Response
"""


OPENAPI_INFO_PROMPT = """
Possibly relevant openapi info:
{openapi_info}
"""

REWARD_PROMPT = """
score the following code snippet in {lang} with the instructions: {instructions}
{code}
"""


class WMDatasetSample(TypedDict):
    instructions: str
    lang: str
    resource_type: str | None
    resource_type_def: str | None
    code: str


class ResourceType(TypedDict):
    resource_type: str
    resource_type_def: str
    lang: str


def get_random_resource_types(
    resource_type_name: str | None, lang: str, resource_types: list[ResourceType], n=2
) -> list[str]:
    selected_resource_types = []
    max_iters = 100
    while len(selected_resource_types) < n and max_iters > 0:
        r = random.choice(resource_types)
        if (
            r["resource_type"]
            != resource_type_name  # not the same resource type as the sample
            and r["lang"] == lang  # same language as the sample
            and len(
                [
                    srt
                    for srt in selected_resource_types
                    if srt["resource_type"] == r["resource_type"]
                ]
            )
            == 0  # not already selected (it's possible to have multiple resource types with the same name)
            and r["resource_type_def"] is not None
        ):
            selected_resource_types.append(r)
        max_iters -= 1

    if len(selected_resource_types) < n:
        raise ValueError(
            f"Could not find {n} random resource types for {resource_type_name} in {lang}"
        )
    return [r["resource_type_def"] for r in selected_resource_types]


def get_resource_types_str(sample):
    rts = []
    if sample["resource_type"] is not None and sample["resource_type_def"] is not None:
        rts = get_random_resource_types(
            sample["resource_type"], sample["lang"], resource_types
        )
        rts.append(sample["resource_type_def"])
    else:
        rts = get_random_resource_types(
            sample["resource_type"], sample["lang"], resource_types, n=3
        )
    random.shuffle(rts)
    resource_types_str = "\n".join(rts)
    return resource_types_str


def load_resource_types():
    with open("data/processed/synthetic_resource_types_v2.yaml", "r") as f:
        synthetic_resource_types = yaml.safe_load(f)
        with open("data/processed/hub_resource_types_v2.yaml", "r") as f:
            hub_resource_types = yaml.safe_load(f)
            return hub_resource_types + synthetic_resource_types


def format_info(openapi_info: dict):
    """Recursively iterate through the openapi info and truncate any string values that are too long"""
    MAX_NESTED_LEVEL = 3
    MAX_CHARS_PER_PATH = 300

    def format_property(property, required=False, level=0):
        if level > MAX_NESTED_LEVEL:
            return {}

        special_keys = ["allOf", "anyOf", "oneOf", "enum"]
        for key in special_keys:
            if key in property:
                if key == "enum":
                    return {"$enum": [p for p in property[key]]}
                elif key == "allOf":
                    result = {}
                    for p in property[key]:
                        res = format_property(p, False, level)
                        if isinstance(res, str):
                            # edge case where the property is a string
                            result = res
                        else:
                            result = {**result, **res}
                    return result
                else:
                    dollar_key = "$" + key
                    return {
                        dollar_key: [
                            format_property(p, False, level) for p in property[key]
                        ]
                    }

        if "properties" not in property and "type" not in property:
            return {}

        if "properties" in property or property["type"] == "object":
            if "properties" not in property:
                return {}
            result = {}
            for k, v in property["properties"].items():
                result[k] = format_property(
                    v, "required" in property and k in property["required"], level + 1
                )
            return result
        elif property["type"] == "array":
            if "items" not in property:
                return {}
            return {"$array": format_property(property["items"], False, level + 1)}
        else:
            if required:
                return property["type"] + "!"
            else:
                return property["type"]

    def format_schemas(method, info):
        requestBody = {}
        response = {}
        if "requestBody" in info:
            body = info["requestBody"]
            if "content" in body:
                for content_type, content in body["content"].items():
                    if "schema" in content:
                        requestBody[content_type] = format_property(content["schema"])

        if method == "get":
            responses = info["responses"]
            keys = list(responses.keys())
            keys_200 = [k for k in keys if k.startswith("2")]

            if len(keys_200) > 0:
                resp_200 = responses[keys_200[0]]
                if "content" in resp_200:
                    for content_type, content in resp_200["content"].items():
                        if "schema" in content:
                            response[content_type] = format_property(content["schema"])

        return requestBody, response

    security = []
    if "security" in openapi_info and openapi_info["security"] is not None:
        for sec in openapi_info["security"]:
            for key in sec.keys():
                if key not in security:
                    security.append(key)
    formatted = {
        "title": openapi_info["info"]["title"],
        "security": security,
        "servers": [server["url"] for server in openapi_info["servers"]],
    }

    paths = []

    for path, methods in openapi_info["paths"].items():
        method, info = list(methods.items())[0]

        # group parameters per type
        parameters = {}
        if "parameters" in info:
            for parameter in info["parameters"]:
                if parameter["in"] not in parameters:
                    parameters[parameter["in"]] = {}
                parameters[parameter["in"]][parameter["name"]] = format_property(
                    parameter["schema"],
                    "required" in parameter and parameter["required"],
                )

        requestBody, response = format_schemas(method, info)

        if "summary" not in info:
            if "operationId" in info:
                info["summary"] = operationId_to_summary(info["operationId"])
            else:
                info["summary"] = ""

        formatted_path = {
            path: {
                method: {
                    "summary": info["summary"],
                }
            }
        }

        if parameters != {}:
            formatted_path[path][method]["parameters"] = parameters

        if requestBody != {}:
            formatted_path[path][method]["requestBody"] = requestBody

        if response != {}:
            formatted_path[path][method]["response"] = response

        formatted_path_str = yaml.dump(formatted_path, sort_keys=False)
        if len(formatted_path_str) > MAX_CHARS_PER_PATH:
            formatted_path_str = formatted_path_str[:MAX_CHARS_PER_PATH] + "...\n"
        paths.append(formatted_path_str)

    metadata = yaml.dump(formatted, sort_keys=False)

    result = metadata + "".join(paths)

    return result


def get_openapi_infos_str(sample, openapi_scripts: list[list]):
    if "openapi_info" not in sample:
        random_script = random.sample(openapi_scripts, 1)[0]
        app = random_script["app"]
        openapi_info = random_script["openapi_info"].copy()
    else:
        app = sample["app"]
        openapi_info = sample["openapi_info"].copy()

    app_openapi_scripts = [script for script in openapi_scripts if script["app"] == app]
    random.shuffle(app_openapi_scripts)
    other_scripts = app_openapi_scripts[:2]
    all_paths = list(openapi_info["paths"].items())
    for script in other_scripts:
        all_paths.append(list(script["openapi_info"]["paths"].items())[0])
    random.shuffle(all_paths)
    openapi_info["paths"] = dict(all_paths)

    openapi_info_str = format_info(openapi_info)

    return openapi_info_str


resource_types: list[ResourceType] = load_resource_types()


def prepare_prompt(
    sample: WMDatasetSample, model_name: str, openapi_scripts: list[dict] | None = None
) -> str:
    kind = "gen"

    if sample["id"].startswith("hubfix_"):
        kind = "fix"
    elif sample["id"].startswith("hubedit_"):
        kind = "edit"

    if kind == "gen":
        resource_types_str = get_resource_types_str(sample)
    else:
        resource_types_str = ""

    if openapi_scripts is not None and kind == "gen":
        openapi_info_str = get_openapi_infos_str(sample, openapi_scripts)
    else:
        openapi_info_str = ""

    if (
        model_name == "salesforce/codet5p-220m"
        or model_name == "salesforce/codet5p-770m"
    ):
        if kind == "edit":
            prompt = SIMPLE_EDIT_PROMPT
        elif kind == "fix":
            prompt = SIMPLE_FIX_PROMPT
        elif kind == "gen":
            prompt = SIMPLE_PROMPT
        else:
            raise ValueError(f"Unknown kind: {kind}")
    elif model_name == "ise-uiuc/Magicoder-S-DS-6.7B":
        if kind == "edit":
            prompt = MAGICODER_EDIT_PROMPT
        elif kind == "fix":
            prompt = MAGICODER_FIX_PROMPT
        elif kind == "gen":
            prompt = MAGICODER_PROMPT
        else:
            raise ValueError(f"Unknown kind: {kind}")
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    prompt = prompt.replace(
        "{lang}",
        sample["lang"],
    )

    if kind == "gen":
        prompt = prompt.replace("{resource_types}", resource_types_str).replace(
            "{instructions}", sample["instructions"]
        )

        if openapi_scripts is not None:
            prompt = prompt.replace(
                "{openapi_info_prompt}",
                OPENAPI_INFO_PROMPT.replace("{openapi_info}", openapi_info_str),
            )
        else:
            prompt = prompt.replace("{openapi_info_prompt}", "")
    if kind == "edit":
        prompt = prompt.replace("{code}", sample["original_code"]).replace(
            "{instructions}", sample["edit_instructions"]
        )
    elif kind == "fix":
        prompt = prompt.replace("{code}", sample["broken_code"]).replace(
            "{error}", sample["error"]
        )

    return prompt


def prepare_label(sample: WMDatasetSample) -> str:
    kind = "gen"

    if sample["id"].startswith("hubfix_"):
        kind = "fix"
    elif sample["id"].startswith("hubedit_"):
        kind = "edit"

    if kind == "gen":
        label = sample["code"]
    elif kind == "edit":
        label = sample["modified_code"]
    elif kind == "fix":
        label = sample["original_code"]
    else:
        raise ValueError(f"Unknown kind: {kind}")

    return label
