from tqdm import tqdm
import json
import re
import yaml


from .utils import generate_sample, parse_code, yaml_dump
from .benchmark import (
    test_mbpp_sample,
    test_ts_mbpp_sample,
)

PROMPT = """
Return the following code with parameter types:
```python
{code}
```
Return a code block with the complete updated code.
"""


def add_typing(code: str):
    response = generate_sample(
        [
            {
                "role": "user",
                "content": PROMPT.format(code=code),
            }
        ]
    )

    code = parse_code(response)

    if code == "":
        raise Exception("Did not find code in final prompt")
    else:
        last_function_name = re.findall(r"def (\S+)\(", code)[-1]
        code = code.replace(last_function_name, "main")
        return code


def add_typing_to_mbpp():
    with open("./data/raw/sanitized-mbpp.json", "r") as f:
        data = json.load(f)
        new_samples = []
        print("Number of samples:", len(data))
        for sample in tqdm(data):
            sample["code"] = add_typing(sample["code"])
            sample["prompt"] = re.sub(
                "Write a (?:python )?function (?:to|that) ", "", sample["prompt"]
            )
            sample["prompt"] = re.sub("\.$", "", sample["prompt"])
            new_samples.append(sample)
            with open("./data/processed/mbpp_typed.yaml", "w") as f:
                yaml_dump(new_samples, f)


def rename_mbpp_tests():
    with open("./data/processed/mbpp_typed.yaml", "r") as f:
        typed_data = yaml.load(f, Loader=yaml.FullLoader)
        with open("./data/raw/sanitized-mbpp.json", "r") as f:
            raw_data = json.load(f)
            new_samples = []
            print("Number of samples:", len(typed_data))
            for sample in tqdm(typed_data):
                raw_code = [
                    raw_sample["code"]
                    for raw_sample in raw_data
                    if raw_sample["task_id"] == sample["task_id"]
                ][0]
                if raw_code is None:
                    print(f"Could not find raw code for sample {sample['task_id']}")
                    continue
                functions = re.findall(r"def (\S+)\(", raw_code)
                if len(functions) == 0:
                    print(f"Could not find functions for sample {sample['task_id']}")
                    continue
                last_function_name = functions[-1]
                sample["test_list"] = [
                    test.replace(last_function_name, "main")
                    for test in sample["test_list"]
                ]
                new_samples.append(sample)
                with open("./data/processed/mbpp_typed.yaml", "w") as f:
                    yaml_dump(new_samples, f)


def filter_valid_mbpp():
    with open("./data/processed/mbpp_typed.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        incorrect_scripts = 0
        valid_samples = []
        for sample in tqdm(data):
            failed = test_mbpp_sample(sample["code"], sample["test_list"])
            if failed:
                incorrect_scripts += 1
            else:
                valid_samples.append(sample)

        with open("./data/processed/mbpp_typed_valid.yaml", "w") as f:
            yaml_dump(valid_samples, f)

        print("Incorrect scripts:", incorrect_scripts, "/", len(data))


def filter_valid_ts_mbpp():
    with open("./data/processed/mbpp_typed_ts.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        incorrect_scripts = 0
        valid_samples = []
        for sample in tqdm(data):
            failed = test_ts_mbpp_sample(sample["code"], sample["test_list"])
            if failed:
                incorrect_scripts += 1
            else:
                valid_samples.append(sample)

        with open("./data/processed/mbpp_typed_ts_valid.yaml", "w") as f:
            yaml_dump(valid_samples, f)

        print("Incorrect scripts:", incorrect_scripts, "/", len(data))


if __name__ == "__main__":
    filter_valid_mbpp()
    filter_valid_ts_mbpp()
