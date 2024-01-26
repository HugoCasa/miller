from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
import re
import yaml
from torch import Tensor


# OpenAI
client = OpenAI()


def generate_sample(
    messages: list[dict[str, str]],
    max_tokens: int = 2048,
    temperature: float = 0,
    top_p: float = 1,
    model_name: str = "gpt-4-1106-preview",
    return_probs: bool = False,
    seed: int | None = 42,
):
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        extra_body={
            "logprobs": True,
            "top_logprobs": 5,
        }
        if return_probs
        else None,
    )
    print(response.usage)
    answer = response.choices[0].message.content
    if return_probs:
        response = response.model_dump(exclude_unset=True)
        return answer, response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return response.choices[0].message.content


# ML


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor | None) -> Tensor:
    if attention_mask is None:
        return last_hidden_states.mean(dim=1)
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# prompts


def parse_code(answer: str):
    matches = re.findall(r"```[a-zA-Z]+\n([\s\S]*?)\n```", answer)
    if len(matches) == 0:
        return ""
    else:
        return matches[-1]


if __name__ == "__main__":
    parse_code("asd")


# resource types


def ts_compile(schema: dict[str, str]) -> str:
    def rec(x, root=False):
        res = "{\n"
        entries = x.items()
        if len(entries) == 0:
            return "any"
        i = 0
        for name, prop in entries:
            if "type" not in prop:
                res += f"  {name}: any"
            elif prop["type"] == "object":
                res += f"  {name}: {rec(prop.get('properties', {}))}"
            elif prop["type"] == "array":
                res += f"  {name}: {prop.get('items', {}).get('type', 'any')}[]"
            else:
                typ = prop.get("type", "any")
                if typ == "integer":
                    typ = "number"
                res += f"  {name}: {typ}"
            i += 1
            if i < len(entries):
                res += ",\n"
        res += "\n}"
        return res

    return rec(schema["properties"], True)


def python_compile(schema: dict[str, str]):
    res = ""
    entries = schema["properties"].items()
    if len(entries) == 0:
        return "dict"
    i = 0
    for name, prop in entries:
        typ = "dict"
        if prop["type"] == "array":
            typ = "list"
        elif prop["type"] == "string":
            typ = "str"
        elif prop["type"] == "number":
            typ = "float"
        elif prop["type"] == "integer":
            typ = "int"
        elif prop["type"] == "boolean":
            typ = "bool"
        res += f"    {name}: {typ}"
        i += 1
        if i < len(entries):
            res += "\n"
    return res


def get_resource_type_def(
    resource_type: str | None, code: str, lang: str
) -> str | None | tuple[str | None, str | None]:
    if resource_type is None:
        match = (
            re.search(
                rf"type (\S+) = {{.*?^}}",
                code,
                re.MULTILINE | re.DOTALL,
            )
            if lang == "deno"
            else re.search(
                rf"class (\S+)\(TypedDict\):.*?\n[^ ]",
                code,
                re.MULTILINE | re.DOTALL,
            )
        )
        return (match.group(1), match.group(0).strip()) if match else (None, None)

    else:
        match = (
            re.search(
                rf"type {resource_type} = {{.*?^}}",
                code,
                re.MULTILINE | re.DOTALL,
            )
            if lang == "deno"
            else re.search(
                rf"class {resource_type}\(TypedDict\):.*?\n[^ ]",
                code,
                re.MULTILINE | re.DOTALL,
            )
        )
        return match.group(0).strip() if match else None


def to_camel_case(snake_str: str):
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def to_pascal_case(snake_str: str):
    components = snake_str.split("_")
    return "".join(x.title() for x in components)


# YAML


class Literal(str):
    pass


def literal_presenter(dumper: yaml.Dumper, data: any):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(Literal, literal_presenter)


def format_literal(answer: str):
    return re.sub("[^\\S\n]+\n", "\n", answer).replace("\t", "    ")


def yaml_dump(data: any, file=None):
    def rec(x):
        if isinstance(x, dict):
            return {k: rec(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [rec(v) for v in x]
        elif isinstance(x, str):
            return Literal(format_literal(x))
        else:
            return x

    yaml.dump(rec(data), file, sort_keys=True)


def remove_markdown_links(text: str):
    return re.sub(r"\.? ?\[[^\]]+\]\([^)]+\)", "", text)


# OpenAPI


def snake_case(name: str):
    return re.sub("([A-Z])", "_\\1", name[0].lower() + name[1:]).lower()


def operationId_to_summary(operationId: str):
    summary = snake_case(operationId).replace("_", " ")

    # capitaliuze first letter
    summary = summary[0].upper() + summary[1:]
    return summary


if __name__ == "__main__":
    print(operationId_to_summary("GetApplicationFees"))
