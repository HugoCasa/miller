import yaml
import random
from tqdm import tqdm

from .utils import generate_sample, parse_code, yaml_dump
from .synthetic_data_gen import (
    GEN_SYSTEM_PROMPT_DENO,
    GEN_SYSTEM_PROMPT_PYTHON,
    GEN_PROMPT,
)


def generate_pairs():
    with open("./data/processed/wm_dataset_v2.yaml", "r") as f:
        wm_dataset = yaml.load(f, Loader=yaml.FullLoader)

    synthetic = [
        script for script in wm_dataset if script["id"].startswith("synthetic_")
    ]

    random.seed(42)
    random.shuffle(synthetic)

    synthetic = synthetic[:1000]

    pairs = []
    for script in tqdm(synthetic):
        prompt = GEN_PROMPT.format(
            name=script["instructions"], integration=script["resource_type"]
        )
        response = generate_sample(
            [
                {
                    "role": "system",
                    "content": GEN_SYSTEM_PROMPT_PYTHON
                    if script["lang"] == "python"
                    else GEN_SYSTEM_PROMPT_DENO,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            seed=None,  # we don't want to use the same seed as when generated the first time
        )

        code = parse_code(response)

        script["code_bis"] = code

        pairs.append(script)

        with open("./data/processed/wm_pairs.yaml", "w") as f:
            yaml_dump(pairs, f)


if __name__ == "__main__":
    generate_pairs()
