import yaml
from tqdm import tqdm
import numpy as np
import random
import os

from .utils import generate_sample, yaml_dump

REWARD_PROMPT = """
Below is two code segments generated from some given instructions. 

Instructions:
{instructions}

Code segment 1: 
```{lang}
{code1}
```

Code segment 2:
```{lang}
{code2}
```

Given code segment 1 and code segment 2, determine which is better.

{lang_context}

You should answer the following questions to determine the best segment:
- Does the code segment work?
- Does the code segment do what is expected?
- Does the code segment contain placeholder code? The code segment should not contain placeholder code and should be complete.
- Is the code segment readable?
- Is the code segment efficient?
- Are edge cases handled?

Consider the instructions, the code segments, the questions and explain which code segment is best and why.
If given the questions, you cannot determine which code segment is best, choose the one you prefer and explain why.
"""

SCORE_PROMPT = """
So given your answer, choose which code segment is better or preferred by returning 1 or 2.
You have to choose one.
Return only the number.
"""


def score_pairs(code1, code2, instructions, lang):
    lang_context = ""
    if lang == "deno":
        lang_context = "As we are using Deno, libraries can be imported from http urls or from npm using the `npm:` prefix. Do not penalize it."
    prompt = REWARD_PROMPT.format(
        lang=lang,
        instructions=instructions,
        code1=code1,
        code2=code2,
        lang_context=lang_context,
    )
    response = generate_sample(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )

    print(response)

    score_response = generate_sample(
        messages=[
            {
                "role": "assistant",
                "content": response,
            },
            {"role": "user", "content": SCORE_PROMPT},
        ],
    )

    score_response = score_response.strip()

    print(score_response)

    if score_response == "1":
        return 1
    elif score_response == "2":
        return 2
    else:
        return None


def score_dataset():
    with open("data/processed/wm_pairs.yaml", "r") as f:
        wm_pairs = yaml.load(f, Loader=yaml.FullLoader)

    scored = []
    existing = None
    if os.path.isfile("data/processed/wm_pairs_scored.yaml"):
        with open("data/processed/wm_pairs_scored.yaml", "r") as f:
            existing = yaml.load(f, Loader=yaml.FullLoader)
            scored = existing

    random.seed(42)
    random.shuffle(wm_pairs)

    skipped = 0
    i = 0
    for sample in tqdm(wm_pairs):
        if existing is not None:
            if any(
                [
                    sample["instructions"] == s["instructions"]
                    and sample["lang"] == s["lang"]
                    for s in existing
                ]
            ):
                i += 1
                print("Already scored, skipping...", i)
                continue

        code1, code2 = sample["code"], sample["code_bis"]
        if random.random() < 0.5:
            code1, code2 = code2, code1
        result = score_pairs(code1, code2, sample["instructions"], sample["lang"])
        if result is None:
            print("Invalid response, skipping...")
            skipped += 1
            print(f"Skipped: {skipped}")
            continue
        good = code1 if result == 1 else code2
        bad = code2 if result == 1 else code1

        sample.pop("code", None)
        sample.pop("code_bis", None)

        scored.append(
            {
                **sample,
                "good": good,
                "bad": bad,
            }
        )

        with open("data/processed/wm_pairs_scored.yaml", "w") as f:
            yaml_dump(scored, f)


if __name__ == "__main__":
    score_dataset()
