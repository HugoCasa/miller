from .utils import generate_sample, parse_code, yaml_dump
from tqdm import tqdm
import yaml
import random
import re
import os

EDIT_INVENT_PROMPT_DENO = """
You are given the following typescript (deno runtime) script:

```typescript
{code}
```

Your task is to invent a modification that you could make to the script that would make it do something different.

Return only the modification instructions, not the code.
The instructions should be short like one or two sentences. It should only include the high-level idea of the modification, not how to implement it.

Wrap your instructions inside <instructions> and </instructions> tags.
"""


EDIT_PROMPT_DENO = """
You are given the following typescript (deno runtime) script:

```typescript
{code}
```


Your task is to edit the script according to this prompt:
{prompt}


Return only a code block.
"""


EDIT_INVENT_PROMPT_PYTHON = """
You are given the following python script:

```python
{code}
```

Your task is to invent a modification that you could make to the script that would make it do something different.

Return only the modification instructions, not the code. 
The instructions should be short like one or two sentences. It should only include the high-level idea of the modification, not how to implement it.

Wrap your instructions inside <instructions> and </instructions> tags.
"""

EDIT_PROMPT_PYTHON = """
You are given the following python script:

```python
{code}
```

Your task is to edit the script according to this prompt:
{prompt}


Return only a code block.
"""

BREAK_PROMPT_DENO = """
Below is a typescript (deno runtime) script that works. Change it so that it breaks.

```typescript
{code}
```

Here are some examples of breaking changes you could introduce:
- importing a library that doesn't exist
- renaming the main function to something else than `main`
- calling a function that doesn't exist
- passing incorrect parameters to a function
- passing the wrong type of parameters to a function
- using a variable that doesn't exist
- using a variable before it is defined
- changing the name of a defined type (e.g. `type Slack` => `type Slacks`)
- using a non-existent property of a type inside the function
- changing the return type of the function
- changing the type of a parameter
- changing the type of a variable
- calling an API method that doesn't exist
- calling an API method with the wrong parameters

The change must be an error that a user would make, do not use placeholder names like `foo` or `bar`.
You can introduce more than one breaking change if you want.

You need to only return three things:
- a code block with the broken code (do not include any comments indicating what was changed)
- the error message that would be printed if you ran the broken code inside <error> and </error> tags
- the explanation of what is wrong with the code inside <explanation> and </explanation> tags
"""

BREAK_PROMPT_PYTHON = """
Below is a python script that works. Change it so that it breaks.

```python
{code}
```

Here are some examples of breaking changes you could introduce:
- importing a library that doesn't exist
- renaming the main function to something else than `main`
- calling a function that doesn't exist
- passing incorrect parameters to a function
- passing the wrong type of parameters to a function
- using a variable that doesn't exist
- using a variable before it is defined
- changing the name of a defined typeddict (e.g. `class slack(TypedDict)` => `class slack_resource(TypedDict)`)
- using a non-existent property of a typeddict inside the function
- changing the return type of the function
- changing the type of a parameter
- changing the type of a variable
- calling an API method that doesn't exist
- calling an API method with the wrong parameters


You can introduce more than one breaking change if you want.

You need to only return two things:
- a code block with the broken code (do not include any comments indicating what was changed)
- the error message that would be printed if you ran the broken code inside <error> and </error> tags
- the explanation of what is wrong with the code inside <explanation> and </explanation> tags
"""


def generate_edit_script(code: str, lang: str):
    # first step
    prompt = EDIT_INVENT_PROMPT_DENO if lang == "deno" else EDIT_INVENT_PROMPT_PYTHON
    draft = generate_sample(
        [
            {
                "role": "user",
                "content": prompt.replace("{code}", code),
            },
        ]
    )

    match = re.search("<instructions>(.+)</instructions>", draft, re.DOTALL)
    if match:
        instructions = match.group(1).strip()
    else:
        raise Exception("Did not find instructions in draft answer")

    prompt = EDIT_PROMPT_DENO if lang == "deno" else EDIT_PROMPT_PYTHON

    print("code:", code)
    print("instructions:", instructions)

    final = generate_sample(
        [
            {
                "role": "user",
                "content": prompt.replace("{code}", code).replace(
                    "{prompt}", instructions
                ),
            },
        ]
    )

    code = parse_code(final)

    if code == "":
        raise Exception("Did not find code")
    else:
        return instructions, code


def generate_fix_script(code: str, lang: str):
    # first step
    prompt = BREAK_PROMPT_DENO if lang == "deno" else BREAK_PROMPT_PYTHON
    answer = generate_sample(
        [
            {
                "role": "user",
                "content": prompt.replace("{code}", code),
            },
        ]
    )

    broken_code = parse_code(answer)

    match = re.search("<explanation>(.+)</explanation>", answer, re.DOTALL)

    if match is None:
        raise Exception("Did not find explanation in answer")
    else:
        explanation = match.group(1).strip()

    match2 = re.search("<error>(.+)</error>", answer, re.DOTALL)

    if match2 is None:
        raise Exception("Did not find error in answer")
    else:
        error = match2.group(1).strip()

    if broken_code == "":
        raise Exception("Did not find code")
    else:
        return explanation, error, broken_code


def generate_edit_scripts():
    with open("./data/processed/wm_dataset_v2.yaml", "r") as f:
        scripts = yaml.load(f, Loader=yaml.FullLoader)

    if "hubedit_generated.yaml" in os.listdir("./data/synthetic_data"):
        with open("./data/synthetic_data/hubedit_generated.yaml", "r") as f:
            generated_scripts = yaml.load(f, Loader=yaml.FullLoader)
            print("loaded", len(generated_scripts), "generated scripts")
    else:
        generated_scripts = []

    scripts = [
        script for script in scripts if not script["id"].startswith("synthetic_")
    ]

    for script in tqdm(scripts):
        if any(
            [
                "hubedit_" + script["id"] == generated_script["id"]
                for generated_script in generated_scripts
            ]
        ):
            print("skipping", script["id"])
            continue
        instructions, code = generate_edit_script(script["code"], script["lang"])
        generated_scripts.append(
            {
                "original_code": script["code"],
                "modified_code": code,
                "lang": script["lang"],
                "edit_instructions": instructions,
                "id": "hubedit_" + script["id"],
                "original_instructions": script["instructions"],
                "resource_type": script["resource_type"],
                "resource_type_def": script["resource_type_def"]
                if "resource_type_def" in script
                else None,
            }
        )
        with open(f"./data/synthetic_data/hubedit_generated.yaml", "w") as f:
            yaml_dump(generated_scripts, f)


def generate_fix_scripts():
    with open("./data/processed/wm_dataset_v2.yaml", "r") as f:
        scripts = yaml.load(f, Loader=yaml.FullLoader)

    generated_scripts = []

    scripts = [
        script for script in scripts if not script["id"].startswith("synthetic_")
    ]

    for script in tqdm(scripts):
        explanation, error, code = generate_fix_script(script["code"], script["lang"])
        generated_scripts.append(
            {
                "original_code": script["code"],
                "broken_code": code,
                "lang": script["lang"],
                "error": error,
                "explanation": explanation,
                "id": "hubfix_" + script["id"],
                "original_instructions": script["instructions"],
                "resource_type": script["resource_type"],
                "resource_type_def": script["resource_type_def"]
                if "resource_type_def" in script
                else None,
            }
        )
        with open(f"./data/synthetic_data/hubfix_generated.yaml", "w") as f:
            yaml_dump(generated_scripts, f)


if __name__ == "__main__":
    # generate_edit_scripts()
    generate_fix_scripts()
