import requests
import base64
import re
from tqdm import tqdm
import yaml
import os
import dotenv

dotenv.load_dotenv()

print(os.environ["GITHUB_TOKEN"])

headers = {
    "Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
}


def get_rate_limit():
    url = "https://api.github.com/rate_limit"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        return content
    else:
        print(f"Failed to get rate limit. Status code: {response.status_code}")


def list_integrations():
    url = f"https://api.github.com/repos/PipedreamHQ/pipedream/contents/components"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        contents = response.json()
        folders = [content["name"] for content in contents if content["type"] == "dir"]
        return folders
    else:
        print(f"Failed to list folders. Status code: {response.status_code}")


def get_scripts(integration: str):
    url = f"https://api.github.com/repos/PipedreamHQ/pipedream/contents/components/{integration}/actions"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        contents = response.json()
        actions = [
            content["name"]
            for content in contents
            if content["type"] == "dir" and content["name"] != "common"
        ]
        for action in tqdm(actions):
            url = f"https://api.github.com/repos/PipedreamHQ/pipedream/contents/components/{integration}/actions/{action}/{action}.mjs"
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                response = requests.get(
                    f"https://api.github.com/repos/PipedreamHQ/pipedream/contents/components/{integration}/actions/{action}/{action}.js",
                    headers=headers,
                )
            if response.status_code == 200:
                content = response.json()
                script = base64.b64decode(content["content"]).decode("utf-8")
                match = re.search(r"name: \"(.+)\"", script)
                if match is None:
                    print(f"Failed to get name of action {action}")
                    continue
                name = match.group(1)
                match = re.search(r"description: \"(.+)\"", script)
                if match is None:
                    print(f"Failed to get description of action {action}")
                    continue
                description = match.group(1)
                yield {
                    "name": name,
                    "description": description,
                    "id": action,
                    "integration": integration,
                }
            else:
                print(
                    f"Failed to get action {action}. Status code: {response.status_code}"
                )
    else:
        print(
            f"Failed to get scripts for integration {integration}. Status code: {response.status_code}"
        )


if __name__ == "__main__":
    # print(get_rate_limit())
    integrations = list_integrations()
    print(integrations)
    all_scripts = []
    for integration in tqdm(integrations):
        scripts = list(get_scripts(integration))
        all_scripts.extend(scripts)
        with open("./data/utils/integrations.yaml", "w") as f:
            yaml.dump(all_scripts, f)
