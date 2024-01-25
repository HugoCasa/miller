from .eval import check_syntax, check_resource_type
import yaml


def get_scores(sample):
    syntax_score = check_syntax(sample["final_code"], sample["lang"])
    resource_type_score = check_resource_type(
        sample["final_code"], sample["lang"], sample["resource_type"]
    )
    return syntax_score, resource_type_score


def check_synthetic_data(path: str):
    syntax_count = 0
    resource_type_count = 0
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        for sample in data:
            syntax_score, resource_type_score = get_scores(sample)
            syntax_count += syntax_score
            resource_type_count += resource_type_score
            print(
                f"Id: {sample['id']}, Syntax: {syntax_score}, Resource Type: {resource_type_score}"
            )
        print(f"Average syntax score: {syntax_count / len(data)}")
        print(f"Average resource type score: {resource_type_count / len(data)}")


if __name__ == "__main__":
    check_synthetic_data("./data/generated_data/pipedream_generated_deno_100.yaml")
    check_synthetic_data("./data/generated_data/pipedream_generated_python_100.yaml")
