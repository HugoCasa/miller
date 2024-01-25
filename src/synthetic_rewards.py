import yaml
from tqdm import tqdm
import numpy as np
import random

from .utils import generate_sample, yaml_dump

REWARD_PROMPT = """
Below is a code segment generated from some given instructions. 

Instructions:
{instructions}

Code: 
```{lang}
{code}
```

Give the code segment a score from 1 to 10, where 1 is the worst and 10 is the best.
A code segment that does what is expected should at least get a score of 5.
A segment that doesn't do what is expected should get a score under 5.
A code segment that doesn't work at all should get a score of 1.
A score of 10 means that the code segment works perfectly, is easy to read, and is efficient.
You should penalize placeholder libraries, placeholder api endpoints, or any other placeholder code. The script should be runnable as is.

{lang_context}

You should answer the following questions to determine the score:
- Does the code segment work?
- Does the code segment do what is expected?
- Does the code segment contain placeholder code?
- Does the code segment call the right API endpoints / use the right libraries, if any?
- Does the code segment call the correctly the API endpoints / correctly use the imported libraries according to your knowledge?
- Is the code segment readable?
- Is the code segment efficient?

Consider the instructions, the code, the questions and explain which score you would give and why.
"""

SCORE_PROMPT = """
What is the score of the code segment? (1-10)
Return only the number.
"""


def score_code(code, instructions, lang):
    lang_context = ""
    if lang == "deno":
        lang_context = "As we are using Deno, libraries can be imported from http urls or from npm using the `npm:` prefix. Do not penalize it."
    prompt = REWARD_PROMPT.format(
        lang=lang, instructions=instructions, code=code, lang_context=lang_context
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

    score_response, score_probs = generate_sample(
        messages=[
            {
                "role": "assistant",
                "content": response,
            },
            {"role": "user", "content": SCORE_PROMPT},
        ],
        return_probs=True,
    )

    print(score_response)
    print(score_probs)
    scores = []
    logprobs = []
    for prob in score_probs:
        if prob["token"] in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
            scores.append(int(prob["token"]))
            logprobs.append(prob["logprob"])
    weighted_score = compute_weighted_score(scores, logprobs)

    print(weighted_score)

    return weighted_score


def compute_weighted_score(scores, logprobs):
    # compute softmax
    probs = np.array(logprobs)
    probs = np.exp(probs) / np.exp(probs).sum()
    weighted_score = 0
    for i, score in enumerate(scores):
        weighted_score += probs[i] * score
    weighted_score = weighted_score / 5 - 1
    return weighted_score.item()


def score_dataset():
    with open("data/processed/wm_dataset_v2.yaml", "r") as f:
        wm_dataset = yaml.load(f, Loader=yaml.FullLoader)

    random.seed(42)
    random.shuffle(wm_dataset)

    scored = []
    for sample in tqdm(wm_dataset[:1000]):
        result = score_code(sample["code"], sample["instructions"], sample["lang"])
        scored.append(
            {
                **sample,
                "score": result,
            }
        )

        with open("data/processed/wm_dataset_v2_scored.yaml", "w") as f:
            yaml_dump(scored, f)


def score_broken_dataset():
    with open("data/synthetic_data/hubfix_generated.yaml", "r") as f:
        hubfix = yaml.load(f, Loader=yaml.FullLoader)

    random.seed(42)
    random.shuffle(hubfix)

    scored = []
    for sample in tqdm(hubfix[:500]):
        result = score_code(
            sample["broken_code"], sample["original_instructions"], sample["lang"]
        )
        scored.append(
            {
                **sample,
                "score": result,
            }
        )

        with open("data/synthetic_data/hubfix_generated_scored.yaml", "w") as f:
            yaml_dump(scored, f)


def combine_rm_dataset():
    with open("data/processed/wm_dataset_v2_scored.yaml", "r") as f:
        wm_dataset = yaml.load(f, Loader=yaml.FullLoader)
    with open("data/synthetic_data/hubfix_generated_scored.yaml", "r") as f:
        hubfix = yaml.load(f, Loader=yaml.FullLoader)

    data = wm_dataset + [
        {
            **script,
            "code": script["broken_code"],
            "instructions": script["original_instructions"],
        }
        for script in hubfix
        if script["score"] <= 0
    ]
    with open("data/processed/reward_model_dataset.yaml", "w") as f:
        yaml_dump(data, f)


if __name__ == "__main__":
    # score_dataset()
    # score_broken_dataset()
    combine_rm_dataset()
