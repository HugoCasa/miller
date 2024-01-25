import subprocess
import yaml
import json
from tqdm import tqdm
import re

from .utils import yaml_dump, generate_sample, parse_code

from .prompts import get_resource_types_str


TIMEOUT = 10


# def test_py(code: str):
#     failed = False
#     error = None
#     try:

#         func_timeout(TIMEOUT, exec, args=[code, globals()])
#     except FunctionTimedOut:
#         failed = True
#         error = "Timeout"
#         print("Timeout")
#     except Exception as e:
#         failed = True
#         error = e
#         print("Error:", e)
#     return failed, error


def test_py(code: str):
    failed = False
    error = None
    code = code + "\nprint('done')"
    with open("./tmp.py", "w") as f:
        f.write(code)
    try:
        out = subprocess.run(
            ["python", "./tmp.py"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        if not out.stdout.endswith("done\n"):
            failed = True
            error = out.stderr
            print("Error:", out.stderr)
    except subprocess.TimeoutExpired:
        failed = True
        error = "Timeout"
        print("Timeout")

    subprocess.run(["rm", "./tmp.py"])

    return failed, error


def test_mbpp_sample(code: str, test_list: list[str]) -> bool:
    failed = False
    for test in test_list:
        full_code = code + "\n" + test
        failed, error = test_py(full_code)
    return failed, error


def test_ts(code: str):
    imports = 'import { equal, assert, assertEquals, assertNotEquals } from "https://deno.land/std@0.210.0/assert/mod.ts";\n'
    code = imports + code
    failed = False
    error = None
    with open("./tmp.ts", "w") as f:
        f.write(code)
    try:
        out = subprocess.run(
            ["deno", "run", "-A", "./tmp.ts"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
        if not out.stdout.endswith("done\n"):
            failed = True
            error = out.stderr
            print("Error:", out.stderr)
    except subprocess.TimeoutExpired:
        failed = True
        error = "Timeout"
        print("Timeout")
    subprocess.run(["rm", "./tmp.ts"])
    return failed, error


def test_ts_mbpp_sample(code: str, test_list: list[str]) -> bool:
    failed = False
    for test in test_list:
        full_code = code + "\n" + test + "\nconsole.log('done');"
        failed, error = test_ts(full_code)
    return failed, error


def test_humaneval_sample(code: str, test: str):
    full_code = code + "\n" + test + "\ncheck(main)"
    failed, error = test_py(full_code)
    return failed, error


def test_ts_humaneval_sample(code: str, test: str):
    full_code = code + "\n" + test + "\nawait check(main)\nconsole.log('done');"
    failed, error = test_ts(full_code)
    return failed, error


def test_custom_sample(code: str, test: str, lang: str, resource: dict[str, str]):
    if lang == "python":
        resource_str = "resource = " + json.dumps(resource) + "\n"
        failed, error = test_py(code + "\n" + resource_str + "\n" + test)
    else:
        resource_str = "const resource = " + json.dumps(resource) + ";\n"
        failed, error = test_ts(
            code + "\n" + resource_str + "\n" + test + "\nconsole.log('done');"
        )
    return failed, error


def run_benchmark(generations: list[str]):
    for sample in tqdm(generations):
        # sample needs to have generated code and id
        sample_id = sample["id"]
        kind, _ = sample_id.split("_", 1)
        lang = sample["lang"]
        print(f"Testing {sample_id}")
        if kind == "humaneval":
            sample["failed"], sample["error"] = (
                test_humaneval_sample(sample["response"], sample["test"])
                if lang == "python"
                else test_ts_humaneval_sample(sample["response"], sample["test"])
            )
            sample["kind"] = "humaneval"
        elif kind == "mbpp":
            sample["failed"], sample["error"] = (
                test_mbpp_sample(sample["response"], sample["test_list"])
                if lang == "python"
                else test_ts_mbpp_sample(sample["response"], sample["test_list"])
            )
            sample["kind"] = "mbpp"
        elif kind == "custom":
            sample["failed"], sample["error"] = test_custom_sample(
                sample["response"], sample["test"], lang, sample["resource"]
            )
            sample["kind"] = "custom"
        else:
            raise Exception("Unknown kind")
        if sample["failed"]:
            print(f"Failed {sample_id}")

    # compute stats for each kind and lang + overall
    stats = {}
    for sample in generations:
        kind = sample["kind"]
        lang = sample["lang"]
        failed = sample["failed"]
        if kind not in stats:
            stats[kind] = {}
        if lang not in stats[kind]:
            stats[kind][lang] = {"failed": 0, "total": 0}
        stats[kind][lang]["total"] += 1
        if failed:
            stats[kind][lang]["failed"] += 1

    overall = {}
    for kind in stats:
        kind_failed = 0
        kind_total = 0
        for lang in stats[kind]:
            stats[kind][lang]["score"] = 1 - (
                stats[kind][lang]["failed"] / stats[kind][lang]["total"]
            )
            kind_failed += stats[kind][lang]["failed"]
            kind_total += stats[kind][lang]["total"]

            if lang not in overall:
                overall[lang] = {"failed": 0, "total": 0}
            overall[lang]["failed"] += stats[kind][lang]["failed"]
            overall[lang]["total"] += stats[kind][lang]["total"]

        stats[kind]["all"] = {
            "failed": kind_failed,
            "total": kind_total,
            "score": 1 - (kind_failed / kind_total),
        }

    print("Overall:", overall)

    all_failed = 0
    all_total = 0
    for lang in overall:
        overall[lang]["score"] = 1 - (overall[lang]["failed"] / overall[lang]["total"])
        all_failed += overall[lang]["failed"]
        all_total += overall[lang]["total"]
    overall["all"] = {
        "failed": all_failed,
        "total": all_total,
        "score": 1 - (all_failed / all_total),
    }
    stats["overall"] = overall

    # print stats
    print("Stats:")
    for kind in stats:
        print(f"  {kind}:")
        for lang in stats[kind]:
            print(
                f"    {lang}: {stats[kind][lang]['total'] - stats[kind][lang]['failed']} / {stats[kind][lang]['total']} = {stats[kind][lang]['score']:.2f}"
            )
    return generations, stats


def debug_custom_benchmark():
    with open("./data/processed/custom_benchmark.yaml", "r") as f:
        custom = yaml.load(f, Loader=yaml.FullLoader)
        nb_failed = 0
        for sample in tqdm(custom[29:]):
            failed, _ = test_custom_sample(
                sample["code"], sample["test"], sample["lang"], sample["resource"]
            )
            if failed:
                nb_failed += 1
                print("Failed:", sample["id"])
        print(f"Failed {nb_failed} / {len(custom)}")


def eval_benchmark_samples(samples_path: str):
    with open(samples_path, "r") as f:
        generations = yaml.load(f, Loader=yaml.FullLoader)

        generations = [
            {
                **sample,
                "response": re.sub(
                    r".*@@ Response\n", "", sample["response"], flags=re.DOTALL
                ),
            }
            for sample in generations
        ]

        results, stats = run_benchmark(generations)
        with open(
            samples_path.replace("_samples.yaml", "_results.yaml"),
            "w",
        ) as f:
            yaml_dump(results, f)

        with open(
            samples_path.replace("_samples.yaml", "_stats.yaml"),
            "w",
        ) as f:
            yaml.dump(stats, f)


OEPNAI_SYSTEM_PROMPT = """
You are a helpful coding assistant for Windmill, a developer platform for running scripts. You write code as instructed by the user. Each user message includes some contextual information which should guide your answer.
Only output code. Wrap the code in a code block.
Put explanations directly in the code as comments.

Here's how interactions have to look like:
user: {sample_question}
assistant: ```language
{code}
```
"""

OPENAI_PYTHON_PROMPT = """
<contextual_information>
You have to write a function in Python called "main". Specify the parameter types. Do not call the main function. You should generally return the result.
You can take as parameters resources which are dictionaries containing credentials or configuration information. For Windmill to correctly detect the resources to be passed, the resource type name has to be exactly as specified in the following list:
<resourceTypes>
{resourceTypes}
</resourceTypes>
You need to define the type of the resources that are needed before the main function, but only include them if they are actually needed to achieve the function purpose.
The resource type name has to be exactly as specified (has to be IN LOWERCASE). If the type name conflicts with any imported methods, you have to rename the imported method with the conflicting name.
<contextual_information>
My instructions: {description}
"""

DENO_PYTHON_PROMPT = """
<contextual_information>
You have to write TypeScript code and export a "main" function like this: "export async function main(...)" and specify the parameter types but do not call it. You should generally return the result.
You can import deno libraries or you can also import npm libraries like that: "import ... from "npm:{{package}}";". The fetch standard method is available globally.
You can take as parameters resources which are dictionaries containing credentials or configuration information. For Windmill to correctly detect the resources to be passed, the resource type name has to be exactly as specified in the following list:
<resourceTypes>
{resourceTypes}
</resourceTypes>
You need to define the type of the resources that are needed before the main function, but only include them if they are actually needed to achieve the function purpose.
The resource type name has to be exactly as specified (no resource suffix). If the type name conflicts with any imported methods, you have to rename the imported method with the conflicting name.
</contextual_information>
My instructions: {description}
"""


def generate_openai_samples():
    with open("./data/processed/benchmark_dataset_v2.yaml", "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    results = []
    for sample in tqdm(data):
        rts = get_resource_types_str(sample)
        prompt = (
            OPENAI_PYTHON_PROMPT if sample["lang"] == "python" else DENO_PYTHON_PROMPT
        ).format(resourceTypes=rts, description=sample["instructions"])

        print(prompt)

        response = generate_sample(
            [
                {
                    "role": "system",
                    "content": OEPNAI_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )
        code = parse_code(response)
        print(code)
        if code == "":
            print("Failed to parse code")
            continue
        results.append(
            {
                **sample,
                "openapi_info": None,
                "response": code,
            }
        )

        with open("./data/generated/openapi_benchmark_samples.yaml", "w") as f:
            yaml_dump(results, f)


if __name__ == "__main__":
    # samples_path = (
    #     # "models/salesforce/codet5p-220m_wmv2_local_2e-05_10_no_rag/benchmark_samples.yaml"
    #     # "models/salesforce/codet5p-220m_wmv2_local_2e-05_10_rag/benchmark_samples.yaml"
    #     # "models/salesforce/codet5p-770m_wmv2_dist_2e-05_10_no_rag/benchmark_samples.yaml"
    #     # "models/salesforce/codet5p-770m_wmv2_dist_2e-05_10_rag/benchmark_samples.yaml"
    #     # "models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_no_rag/benchmark_samples.yaml"
    #     # "models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_rag/benchmark_samples.yaml"
    #     "models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_rag_dpo/benchmark_samples.yaml"
    # )

    # samples_path = "models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_rag_4bit/benchmark_samples.yaml"

    paths = [
        "models/salesforce/codet5p-220m_wmv2_local_2e-05_10_no_rag/benchmark_samples.yaml",
        "models/salesforce/codet5p-220m_wmv2_local_2e-05_10_rag/benchmark_samples.yaml",
        "models/salesforce/codet5p-770m_wmv2_dist_2e-05_10_no_rag/benchmark_samples.yaml",
        "models/salesforce/codet5p-770m_wmv2_dist_2e-05_10_rag/benchmark_samples.yaml",
        "models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_no_rag/benchmark_samples.yaml",
        "models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_rag/benchmark_samples.yaml",
        "models/ise-uiuc/Magicoder-S-DS-6.7B_wmv2_dist_2e-05_10_rag_dpo/benchmark_samples.yaml",
    ]

    for path in paths:
        eval_benchmark_samples(path)

    # samples_path = "models/openai/gpt4turbo/benchmark_samples.yaml"

    # eval_benchmark_samples(samples_path)

    # debug_custom_benchmark()

    # generate_openai_samples()
