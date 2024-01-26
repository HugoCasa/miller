from tqdm import tqdm
import yaml
from src.eval import check_syntax, check_resource_type, check_main_func
from src.data_prep import get_resource_type_def, remove_markdown_links


synthetic_data_paths = [
    "./data/synthetic_data/pipedream_generated_deno_100.yaml",
    "./data/synthetic_data/pipedream_generated_deno_100_1000.yaml",
    "./data/synthetic_data/pipedream_generated_deno_1000_.yaml",
    "./data/synthetic_data/pipedream_generated_python_100.yaml",
    "./data/synthetic_data/pipedream_generated_python_100_1000.yaml",
    "./data/synthetic_data/pipedream_generated_python_1000_.yaml",
]


def eval_synthetic_data():
    # read data
    data = []
    for path in synthetic_data_paths:
        with open(path, "r") as f:
            data.extend(yaml.load(f, Loader=yaml.FullLoader))

    success = 0
    pbar = tqdm(data)
    for sample in pbar:
        code_key = "first_code"
        resource_type_def = get_resource_type_def(
            sample["resource_type"], sample[code_key], sample["lang"]
        )
        if resource_type_def is None:
            sample[code_key] = sample[code_key].replace(
                sample["resource_type"] + "Resource", sample["resource_type"]
            )
        syntax_score = check_syntax(sample[code_key], sample["lang"])
        resource_type_score = (
            check_resource_type(
                sample[code_key], sample["lang"], sample["resource_type"]
            )
            if sample["resource_type"]
            else 1
        )
        main_func_score = check_main_func(sample[code_key], sample["lang"])

        first_step_success = syntax_score and resource_type_score and main_func_score

        second_step_success = False
        if not first_step_success:
            code_key = "final_code"
            resource_type_def = get_resource_type_def(
                sample["resource_type"], sample[code_key], sample["lang"]
            )
            if resource_type_def is None:
                sample[code_key] = sample[code_key].replace(
                    sample["resource_type"] + "Resource", sample["resource_type"]
                )
            syntax_score = check_syntax(sample[code_key], sample["lang"])
            resource_type_score = (
                check_resource_type(
                    sample[code_key], sample["lang"], sample["resource_type"]
                )
                if sample["resource_type"]
                else 1
            )
            main_func_score = check_main_func(sample[code_key], sample["lang"])
            second_step_success = (
                syntax_score and resource_type_score and main_func_score
            )

        success += 1 if first_step_success or second_step_success else 0
        pbar.set_description(f"Successes: {success}")

    print("Success rate:", success / len(data))


if __name__ == "__main__":
    eval_synthetic_data()
