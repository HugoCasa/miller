from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    TextStreamer,
)
import torch
import time
import numpy as np
import requests
import os
import concurrent.futures

client = OpenAI()

DEVICE = torch.device("cuda")


HF_API_TOKEN = os.environ.get("TOKEN")


def time_hf_api(input_text: str, url: str):
    tokenizer = AutoTokenizer.from_pretrained("hugocasa/miller-6.7b-openapi-aligned")
    speeds = []
    for _ in range(1):
        payload = {
            "inputs": input_text,
            "return_full_text": False,
            "parameters": {"max_new_tokens": 512, "do_sample": False},
        }
        start_time = time.time()
        response = requests.post(
            url, json=payload, headers={"Authorization": f"Bearer {HF_API_TOKEN}"}
        )
        end_time = time.time()
        result = response.json()[0]["generated_text"]
        length = len(tokenizer.tokenize(result))
        speeds.append(length / (end_time - start_time))
    print(result)
    print(f"speed: {np.mean(speeds)} tokens/s ({np.std(speeds)})")


def batch_time_hf_api(input_text: str, url: str):
    tokenizer = AutoTokenizer.from_pretrained("hugocasa/miller-6.7b-openapi-aligned")

    def query(payload: dict):
        response = requests.post(
            url, json=payload, headers={"Authorization": f"Bearer {HF_API_TOKEN}"}
        )
        result = response.json()[0]["generated_text"]
        length = len(tokenizer.tokenize(result))
        return length

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        payload = {
            "inputs": input_text,
            "return_full_text": False,
            "parameters": {"max_new_tokens": 512, "do_sample": False},
        }
        futures = [executor.submit(query, payload) for _ in range(8)]

        length = 0
        start_time = time.time()
        for future in concurrent.futures.as_completed(futures):
            l = future.result()
            length += l
        end_time = time.time()
    print(f"speed: {length / (end_time - start_time)} tokens/s")


def time_gpt4turbo_api(messages: str):
    speeds = []
    for _ in range(5):
        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            max_tokens=2048,
            seed=42,
        )
        end_time = time.time()
        length = response.usage.completion_tokens
        speeds.append(length / (end_time - start_time))
    print(response.choices[0].message.content)
    print(f"speed: {np.mean(speeds)} tokens/s ({np.std(speeds)})")


if __name__ == "__main__":
    # HF_API_URL = "https://pglpv7ddht2mzzdv.us-east-1.aws.endpoints.huggingface.cloud"
    HF_API_URL = "https://fqivs97i5mo9fa8c.us-east-1.aws.endpoints.huggingface.cloud"

    text_magicoder = """
    You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

    @@ Instruction
    Possibly relevant resource types:
    class cloudflare_api_key(TypedDict):
        email: str
        api_key: str
    class asana(TypedDict):
        token: str
    class nethunt_crm(TypedDict):
        api_key: str
        base_url: str

    generate the code for the following description in python: return the number of users in a given asana team

    @@ Response
    """
    # time_hf_api(text_magicoder, HF_API_URL)
    batch_time_hf_api(text_magicoder, HF_API_URL)

    system = """
    You are a helpful coding assistant for Windmill, a developer platform for running scripts. You write code as instructed by the user. Each user message includes some contextual information which should guide your answer.
    Only output code. Wrap the code in a code block.
    Put explanations directly in the code as comments.

    Here's how interactions have to look like:
    user: {sample_question}
    assistant: ```language
    {code}
    ```
    """

    user = """
    <contextual_information>
    You have to write TypeScript code and export a "main" function like this: "export async function main(...)" and specify the parameter types but do not call it. You should generally return the result.
    You can import deno libraries or you can also import npm libraries like that: "import ... from "npm:{package}";". The fetch standard method is available globally.
    You can take as parameters resources which are dictionaries containing credentials or configuration information. For Windmill to correctly detect the resources to be passed, the resource type name has to be exactly as specified in the following list:
    <resourceTypes>
    class cloudflare_api_key(TypedDict):
        email: str
        api_key: str
    class asana(TypedDict):
        token: str
    class nethunt_crm(TypedDict):
        api_key: str
        base_url: str
    </resourceTypes>
    You need to define the type of the resources that are needed before the main function, but only include them if they are actually needed to achieve the function purpose.
    The resource type name has to be exactly as specified (no resource suffix). If the type name conflicts with any imported methods, you have to rename the imported method with the conflicting name.
    </contextual_information>
    My instructions: return the number of users in a given asana team
    """
    # time_gpt4turbo_api(
    #     [
    #         {
    #             "content": system,
    #             "role": "system",
    #         },
    #         {
    #             "content": user,
    #             "role": "user",
    #         },
    #     ]
    # )
