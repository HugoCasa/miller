import math
import yaml
import random
from typing import TypedDict
from tqdm import tqdm
import re
import json

from .eval import check_syntax, check_resource_type, check_main_func
from .utils import yaml_dump, remove_markdown_links, get_resource_type_def


class WMDatasetSample(TypedDict):
    id: str
    instructions: str
    code: str
    lang: str
    resource_type: str | None


class WMDatasetSampleWithScores(WMDatasetSample):
    syntax_score: int
    resource_type_score: int
    main_func_score: int


synthetic_data_paths = [
    "./data/synthetic_data/pipedream_generated_deno_100.yaml",
    "./data/synthetic_data/pipedream_generated_deno_100_1000.yaml",
    "./data/synthetic_data/pipedream_generated_deno_1000_.yaml",
    "./data/synthetic_data/pipedream_generated_python_100.yaml",
    "./data/synthetic_data/pipedream_generated_python_100_1000.yaml",
    "./data/synthetic_data/pipedream_generated_python_1000_.yaml",
]


def print_stats(data: list[WMDatasetSampleWithScores], prefix: str | None = None):
    if prefix is not None:
        data = [sample for sample in data if sample["id"].startswith(prefix)]
        print(f"\n---------- Stats for {prefix} ({len(data)} samples) ----------")
    else:
        print(f"\n---------- Stats for all data ({len(data)} samples) ----------")
    # log stats
    correct_syntax_rate = sum([sample["syntax_score"] for sample in data]) / len(data)
    correct_rt_rate = sum([sample["resource_type_score"] for sample in data]) / len(
        data
    )
    correct_mf_rate = sum([sample["main_func_score"] for sample in data]) / len(data)
    correct_rate = sum(
        [
            sample["syntax_score"]
            and sample["resource_type_score"]
            and sample["main_func_score"]
            for sample in data
        ]
    ) / len(data)

    print(f"Average syntax score: {correct_syntax_rate}")
    print(f"Average resource type score: {correct_rt_rate}")
    print(f"Average main func score: {correct_mf_rate}")
    print(f"Average correct score: {correct_rate} ({len(data)})")


def process_data():
    synthetic_data = load_synthetic_data()
    mbpp_data = load_mbpp()
    humaneval_data = load_humaneval()
    hub_scripts = load_hub_scripts()

    data = synthetic_data + mbpp_data + hub_scripts + humaneval_data

    for sample in tqdm(data):
        syntax_score = check_syntax(sample["code"], sample["lang"])
        resource_type_score = (
            check_resource_type(sample["code"], sample["lang"], sample["resource_type"])
            if sample["resource_type"]
            else 1
        )
        main_func_score = check_main_func(sample["code"], sample["lang"])
        sample["syntax_score"] = syntax_score
        sample["resource_type_score"] = resource_type_score
        sample["main_func_score"] = main_func_score

    print_stats(data)
    print_stats(data, "synthetic")
    print_stats(data, "mbpp")
    print_stats(data, "humaneval")
    print_stats(data, "hub")

    # filter data
    filtered_data = [
        sample
        for sample in data
        if sample["syntax_score"]
        and sample["resource_type_score"]
        and sample["main_func_score"]
    ]

    # write to processed
    print(f"Valid data count: {len(filtered_data)}")
    with open("./data/processed/wm_dataset_v2.yaml", "w") as f:
        yaml_dump(filtered_data, f)


def load_synthetic_data() -> list[WMDatasetSample]:
    # read data
    data = []
    for path in synthetic_data_paths:
        with open(path, "r") as f:
            data.extend(yaml.load(f, Loader=yaml.FullLoader))

    resource_types = []
    processed = []
    for sample in tqdm(data):
        resource_type_def = get_resource_type_def(
            sample["resource_type"], sample["final_code"], sample["lang"]
        )
        if resource_type_def is None:
            print("Could not find resource type definition in", sample["id"])
            sample["final_code"] = sample["final_code"].replace(
                sample["resource_type"] + "Resource", sample["resource_type"]
            )
            resource_type_def = get_resource_type_def(
                sample["resource_type"], sample["final_code"], sample["lang"]
            )
            if resource_type_def is None:
                print(
                    "After cleaning, could still not find resource type definition in",
                    sample["id"],
                )

        processed.append(
            {
                "id": "synthetic_" + sample["id"],
                "instructions": remove_markdown_links(sample["description"])
                + " in "
                + sample["integration"].replace("_", " "),
                "code": sample["final_code"],
                "lang": sample["lang"],
                "resource_type": sample["resource_type"],
                "resource_type_def": resource_type_def,
            }
        )

        resource_types.append(
            {
                "resource_type": sample["resource_type"],
                "resource_type_def": resource_type_def,
                "lang": sample["lang"],
            }
        )

    # write resource types to file
    with open("./data/processed/synthetic_resource_types_v2.yaml", "w") as f:
        yaml_dump(resource_types, f)

    return processed


def load_mbpp() -> list[WMDatasetSample]:
    with open("./data/processed/mbpp_typed_valid.yaml", "r") as f:
        py_data = yaml.load(f, Loader=yaml.FullLoader)
        with open("./data/processed/mbpp_typed_ts_valid.yaml", "r") as f:
            ts_data = yaml.load(f, Loader=yaml.FullLoader)
            random.seed(42)
            random.shuffle(py_data)
            split = math.floor(len(py_data) * 0.8)
            py_train_data = py_data[:split]
            py_benchmark_data = py_data[split:]
            ts_benchmark_data = [
                s
                for s in ts_data
                if s["task_id"] in [s["task_id"] for s in py_benchmark_data]
            ]
            ts_train_data = [
                s
                for s in ts_data
                if s["task_id"] in [s["task_id"] for s in py_train_data]
            ]

            processed = []
            benchmark = []
            for sample in tqdm(py_train_data):
                processed.append(
                    {
                        "id": "mbpp_py_" + str(sample["task_id"]),
                        "instructions": sample["prompt"],
                        "code": sample["code"],
                        "lang": "python",
                        "resource_type": None,
                    }
                )
            for sample in tqdm(py_benchmark_data):
                benchmark.append(
                    {
                        "id": "mbpp_py_" + str(sample["task_id"]),
                        "instructions": sample["prompt"],
                        "code": sample["code"],
                        "lang": "python",
                        "resource_type": None,
                        "test_list": sample["test_list"],
                    }
                )

            for sample in tqdm(ts_train_data):
                processed.append(
                    {
                        "id": "mbpp_ts_" + str(sample["task_id"]),
                        "instructions": sample["prompt"],
                        "code": sample["code"],
                        "lang": "deno",
                        "resource_type": None,
                    }
                )
            for sample in tqdm(ts_benchmark_data):
                benchmark.append(
                    {
                        "id": "mbpp_ts_" + str(sample["task_id"]),
                        "instructions": sample["prompt"],
                        "code": sample["code"],
                        "lang": "deno",
                        "resource_type": None,
                        "test_list": sample["test_list"],
                    }
                )
        with open("./data/processed/mbpp_benchmark_v2.yaml", "w") as f:
            yaml_dump(benchmark, f)
        return processed


def load_humaneval() -> list[WMDatasetSample]:
    with open("./data/processed/humaneval_valid.yaml", "r") as f:
        py_data = yaml.load(f, Loader=yaml.FullLoader)
        with open("./data/processed/humaneval_ts_valid.yaml", "r") as f:
            ts_data = yaml.load(f, Loader=yaml.FullLoader)
            random.seed(42)
            random.shuffle(py_data)
            split = math.floor(len(py_data) * 0.5)
            py_train_data = py_data[:split]
            py_benchmark_data = py_data[split:]
            ts_benchmark_data = [
                s
                for s in ts_data
                if s["task_id"] in [s["task_id"] for s in py_benchmark_data]
            ]
            ts_train_data = [
                s
                for s in ts_data
                if s["task_id"] in [s["task_id"] for s in py_train_data]
            ]

            processed = []
            benchmark = []
            for sample in tqdm(py_train_data):
                processed.append(
                    {
                        "id": "humaneval_py_" + str(sample["task_id"]),
                        "instructions": sample["prompt"],
                        "code": sample["code"],
                        "lang": "python",
                        "resource_type": None,
                    }
                )
            for sample in tqdm(py_benchmark_data):
                benchmark.append(
                    {
                        "id": "humaneval_py_" + str(sample["task_id"]),
                        "instructions": sample["prompt"],
                        "code": sample["code"],
                        "lang": "python",
                        "resource_type": None,
                        "test": sample["test"],
                    }
                )

            for sample in tqdm(ts_train_data):
                processed.append(
                    {
                        "id": "humaneval_ts_" + str(sample["task_id"]),
                        "instructions": sample["prompt"],
                        "code": sample["code"],
                        "lang": "deno",
                        "resource_type": None,
                    }
                )
            for sample in tqdm(ts_benchmark_data):
                benchmark.append(
                    {
                        "id": "humaneval_ts_" + str(sample["task_id"]),
                        "instructions": sample["prompt"],
                        "code": sample["code"],
                        "lang": "deno",
                        "resource_type": None,
                        "test": sample["test"],
                    }
                )
        with open("./data/processed/humaneval_benchmark_v2.yaml", "w") as f:
            yaml_dump(benchmark, f)
        return processed


def load_hub_scripts() -> list[WMDatasetSample]:
    with open("./data/raw/hub-scripts.json", "r") as f:
        hub_data = json.load(f)
        with open("./data/raw/hub-scripts-py.yaml", "r") as f:
            py_data = yaml.load(f, Loader=yaml.FullLoader)

            processed = []
            resource_types = []
            ts_data = list(filter(lambda x: x["language"] == "deno", hub_data))
            data = py_data + ts_data
            for sample in tqdm(data):
                resource_type, resource_type_def = get_resource_type_def(
                    None, sample["content"], sample["language"]
                )
                processed.append(
                    {
                        "id": "hub_" + str(sample["script_id"]),
                        "instructions": sample["summary"]
                        + " in "
                        + sample["app"].replace("_", " "),
                        "code": sample["content"],
                        "lang": sample["language"],
                        "resource_type": resource_type,
                        "resource_type_def": resource_type_def,
                    }
                )
                if resource_type is not None:
                    resource_types.append(
                        {
                            "resource_type": resource_type,
                            "resource_type_def": resource_type_def,
                            "lang": sample["language"],
                        }
                    )
                    with open("./data/processed/hub_resource_types_v2.yaml", "w") as f:
                        yaml_dump(resource_types, f)

            return processed


def remove_benchmark_hub_scripts():
    with open("./data/processed/custom_benchmark.yaml", "r") as f:
        custom = yaml.load(f, Loader=yaml.FullLoader)

    hub_ids = ["hub_" + str(s["hub_id"]) for s in custom if "hub_id" in s]

    print(hub_ids)

    with open("./data/processed/wm_dataset_v2.yaml", "r") as f:
        samples = yaml.load(f, Loader=yaml.FullLoader)

    print(f"Removing benchmark scripts from wm_dataset_v2.yaml ({len(samples)})")

    filtered_samples = [sample for sample in samples if sample["id"] not in hub_ids]

    print(f"Remaining samples: {len(filtered_samples)}")

    with open("./data/processed/wm_dataset_v2.yaml", "w") as f:
        yaml_dump(filtered_samples, f)

    with open("./data/processed/wm_dataset_v2_scored.yaml", "r") as f:
        samples = yaml.load(f, Loader=yaml.FullLoader)

    print(f"Removing benchmark scripts from wm_dataset_v2_scored.yaml ({len(samples)})")

    filtered_samples = [sample for sample in samples if sample["id"] not in hub_ids]

    print(f"Remaining samples: {len(filtered_samples)}")

    with open("./data/processed/wm_dataset_v2_scored.yaml", "w") as f:
        yaml_dump(filtered_samples, f)


def remove_too_long_scripts():
    with open("./data/processed/wm_dataset_v2.yaml", "r") as f:
        samples = yaml.load(f, Loader=yaml.FullLoader)

    print(f"Removing benchmark scripts from wm_dataset_v2.yaml ({len(samples)})")

    filtered_samples = [sample for sample in samples if len(sample["code"]) < 1500]

    print(f"Remaining samples: {len(filtered_samples)}")

    with open("./data/processed/wm_dataset_v2.yaml", "w") as f:
        yaml_dump(filtered_samples, f)

    with open("./data/processed/wm_dataset_v2_scored.yaml", "r") as f:
        samples = yaml.load(f, Loader=yaml.FullLoader)

    print(f"Removing benchmark scripts from wm_dataset_v2_scored.yaml ({len(samples)})")

    filtered_samples = [sample for sample in samples if len(sample["code"]) < 1500]

    print(f"Remaining samples: {len(filtered_samples)}")

    with open("./data/processed/wm_dataset_v2_scored.yaml", "w") as f:
        yaml_dump(filtered_samples, f)


def combine_benchmark_dataset():
    with open("./data/processed/custom_benchmark.yaml", "r") as f:
        custom = yaml.load(f, Loader=yaml.FullLoader)
    with open("./data/processed/mbpp_benchmark_v2.yaml", "r") as f:
        mbpp = yaml.load(f, Loader=yaml.FullLoader)
    with open("./data/processed/humaneval_benchmark_v2.yaml", "r") as f:
        humaneval = yaml.load(f, Loader=yaml.FullLoader)

    data = custom + mbpp + humaneval

    with open("./data/processed/benchmark_dataset_v2.yaml", "w") as f:
        yaml.dump(data, f)


if __name__ == "__main__":
    # process_data()
    # remove_benchmark_hub_scripts()
    # remove_too_long_scripts()
    combine_benchmark_dataset()
