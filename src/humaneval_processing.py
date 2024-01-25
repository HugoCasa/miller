from tqdm import tqdm
import yaml
import jsonlines
import re


from .utils import yaml_dump
from .benchmark import (
    test_humaneval_sample,
    test_ts_humaneval_sample,
)


def process_humaneval():
    with jsonlines.open("./data/raw/HumanEval.jsonl", "r") as reader:
        new_samples = []
        for sample in tqdm(reader):
            instr_match = re.search(
                "(?:\"\"\"|''')(.+)(?:\"\"\"|''')", sample["prompt"], re.DOTALL
            )
            if instr_match:
                func_def = sample["prompt"].replace(instr_match.group(0), "")
                sample["prompt"] = instr_match.group(1)
                complete_code = func_def + sample["canonical_solution"]
                complete_code = complete_code.replace(
                    f"def {sample['entry_point']}", "def main"
                )
                sample["code"] = complete_code
                new_samples.append(sample)
                with open("./data/processed/humaneval.yaml", "w") as f:
                    yaml_dump(new_samples, f)
            else:
                print(f"No instructions found for sample {sample['task_id']}")
                continue


def filter_valid_humaneval():
    with open("./data/processed/humaneval.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        incorrect_scripts = 0
        valid_samples = []
        for sample in tqdm(data):
            failed = test_humaneval_sample(sample["code"], sample["test"])
            if failed:
                incorrect_scripts += 1
            else:
                valid_samples.append(sample)

        with open("./data/processed/humaneval_valid.yaml", "w") as f:
            yaml_dump(valid_samples, f)

        print("Incorrect scripts:", incorrect_scripts, "/", len(data))


def filter_valid_ts_humaneval():
    with open("./data/processed/humaneval_ts.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        incorrect_scripts = 0
        valid_samples = []
        for sample in tqdm(data):
            failed = test_ts_humaneval_sample(sample["code"], sample["test"])
            if failed:
                incorrect_scripts += 1
            else:
                valid_samples.append(sample)

        with open("./data/processed/humaneval_ts_valid.yaml", "w") as f:
            yaml_dump(valid_samples, f)
        print("Incorrect scripts:", incorrect_scripts, "/", len(data))


if __name__ == "__main__":
    filter_valid_humaneval()
    filter_valid_ts_humaneval()
