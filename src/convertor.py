from tqdm import tqdm
from .utils import (
    generate_sample,
    parse_code,
    yaml_dump,
    to_pascal_case,
    get_resource_type_def,
)
import json
import yaml

TS2PY_PROMPT = """
Convert the following code to python:
```typescript
{code}
```
It should be exactly the same, except that it shouldn't be async.
The code should be completely functional and runnable. If a libary used doesn't exist in python, you can replace it with a similar library.
The {integration} type should be replaced by a TypedDict with the type name in lowercase.
Only return a code block with the complete python code.
Do not include example usage or any additional comments.
"""


PY2TS_PROMPT = """
Convert the following code to typescript:
```python
{code}
```
It should be exactly the same, except that the main function should be async and exported.
The code should be completely functional and runnable. You cannot import any libraries.
Only return a code block with the complete typescript code.
Do not include example usage or any additional comments.
"""


TEST_PY2TS_PROMPT = """
Convert the following test code to typescript:
```python
{test_code}
```
It should be exactly the same apart that the tested function returns a promise.
Assume the following functions are globally available:
- equal() - Deep comparison function, where actual and expected are compared deeply, and if they vary, equal returns false.
- assert() - Expects a boolean value, throws if the value is false.
- assertEquals() - Uses the equal comparison and throws if the actual and expected are not equal.
- assertNotEquals() - Uses the equal comparison and throws if the actual and expected are equal.
Only return a code block with the typescript code.
Do not include example usage or any additional comments.
"""


def convert_to_python(code: str, integration: str) -> str:
    # integration is in snake case
    ts_app = to_pascal_case(integration)
    code = code.replace(ts_app, integration)

    response = generate_sample(
        [
            {
                "role": "user",
                "content": TS2PY_PROMPT.format(code=code, integration=integration),
            }
        ]
    )

    code = parse_code(response)

    print(code)

    if code == "":
        print(response)
        raise Exception("Did not find code block")
    else:
        return code


def convert_to_ts(code: str) -> str:
    response = generate_sample(
        [
            {
                "role": "user",
                "content": PY2TS_PROMPT.format(code=code),
            }
        ]
    )

    code = parse_code(response)

    print(code)

    if code == "":
        print(response)
        raise Exception("Did not find code block")
    else:
        return code


def convert_test_to_ts(test_code: str) -> str:
    response = generate_sample(
        [
            {
                "role": "user",
                "content": TEST_PY2TS_PROMPT.format(test_code=test_code),
            }
        ]
    )

    code = parse_code(response)

    print(code)

    if code == "":
        print(response)
        raise Exception("Did not find code block")
    else:
        return code


def hub_scripts_to_python():
    with open("./data/raw/hub-scripts.json", "r") as f:
        data = json.load(f)
        data = [sample for sample in data if sample["language"] == "deno"]
        new_samples = []
        print("Number of samples:", len(data))
        for sample in tqdm(data):
            sample["content"] = convert_to_python(sample["content"], sample["app"])
            sample["language"] = "python"
            new_samples.append(sample)
            with open("./data/raw/hub-scripts-py.yaml", "w") as f:
                yaml_dump(new_samples, f)


def mbpp_to_ts():
    with open("./data/processed/mbpp_typed.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        new_samples = []
        print("Number of samples:", len(data))
        for sample in tqdm(data):
            sample["code"] = convert_to_ts(sample["code"])
            sample["test_list"] = [
                convert_test_to_ts(test) for test in sample["test_list"]
            ]
            new_samples.append(sample)
            with open("./data/processed/mbpp_typed_ts.yaml", "w") as f:
                yaml_dump(new_samples, f)


def humaneval_to_ts():
    with open("./data/processed/humaneval.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        new_samples = []
        print("Number of samples:", len(data))
        for sample in tqdm(data):
            sample["code"] = convert_to_ts(sample["code"])
            sample["test"] = convert_test_to_ts(sample["test"])
            new_samples.append(sample)
            with open("./data/processed/humaneval_ts.yaml", "w") as f:
                yaml_dump(new_samples, f)


def openapi_to_python():
    with open("./data/openapi/scripts.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        new_samples = []
        print("Number of samples:", len(data))
        for sample in tqdm(data):
            sample["code"] = convert_to_python(sample["code"], sample["app"])
            sample["lang"] = "python"
            new_samples.append(sample)
            with open("./data/openapi/scripts_py.yaml", "w") as f:
                yaml.dump(new_samples, f)


def openapi_resource_types_to_python():
    with open("data/openapi/scripts_py.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    nones = 0
    new_samples = []
    for sample in data:
        sample["resource_type"], sample["resource_type_def"] = get_resource_type_def(
            None, sample["code"], "python"
        )
        if sample["resource_type"] is None:
            print("None:", sample["id"])
            nodes += 1
        else:
            new_samples.append(sample)

    print("Original nb of samples:", len(data), "Removed (no rt):", nones)

    with open("data/openapi/scripts_py.yaml", "w") as f:
        yaml.dump(new_samples, f)


if __name__ == "__main__":
    # hub_scripts_to_python()
    # mbpp_to_ts()
    # humaneval_to_ts()
    # openapi_to_python()
    openapi_resource_types_to_python()
