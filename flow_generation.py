from llama_cpp import Llama

llm = Llama(model_path="models/mistral/mistral-7b-v0.1.Q5_K_M.gguf", n_gpu_layers=-1)

MISTRAL_PROMPT = """<s>[INST] {prompt} [/INST]"""

MISTRAL_WITH_EXAMPLE = """<s>[INST] {example_prompt} [/INST] {example_answer} </s> [INST] {prompt} [/INST]"""

SCRIPT_PROMPT = """
<contextual_information>
You have to write a function in Python called "main". Specify the parameter types. Do not call the main function. You should generally return the result.
You can take as parameters resources which are dictionaries containing credentials or configuration information. For Windmill to correctly detect the resources to be passed, the resource type name has to be exactly as specified in the following list:
<resourceTypes>
{resource_types}
</resourceTypes>
You need to define the type of the resources that are needed before the main function, but only include them if they are actually needed to achieve the function purpose.
The resource type name has to be exactly as specified (has to be IN LOWERCASE). If the type name conflicts with any imported methods, you have to rename the imported method with the conflicting name.
<contextual_information>
My instructions: {description}
Return a code block.
"""


def generate_script(description: str, resource_types: str):
    prompt = SCRIPT_PROMPT.format(
        description=description, resource_types=resource_types
    )
    mistral_prompt = MISTRAL_PROMPT.format(prompt=prompt)
    print(mistral_prompt)
    response = llm(mistral_prompt, max_tokens=512, temperature=0, top_p=1)
    return response["choices"][0]["text"]


# EXAMPLES = """
# <s>[INST]
# Script a: Get the name and phone number from the given customer id
# Script b: Invent a name for the customer

# Pipe the output of the appropriate script to the following input:
# If none are appropriate, use flow_input.[input_name]
# customer_name=
# [/INST]
# a.name
# </s>
# [INST]
# Script a: Get the number of customers per country

# Pipe the output of the appropriate script to the following input:
# If none are appropriate, use flow_input.[input_name]
# postgresql_config=
# [/INST]
# flow_input.postgresql_config
# </s>
# """

EXAMPLES = """
Script a: Invent a name for the customer
Script d: Get the name and phone number from the given customer id
Pipe the output of the appropriate script to the following input:
If none are appropriate, use flow_input.[input_name]
customer_name=b.name

Script b: Get the weather data for the given city
Script h: Get the number of people in the given city
Pipe the output of the appropriate script to the following input:
If none are appropriate, use flow_input.[input_name]
nb_people=h

Script i: List customers from database
Script k: Send email to the given email address
Pipe the output of the appropriate script to the following input:
If none are appropriate, use flow_input.[input_name]
postgresql_config=flow_input.postgresql_config
"""

LOOP_EXAMPLES = """
Last script: List customers from database
You are in a for loop. You can access one element of the last script's output with the following syntax: flow_input.iter.value
Pipe the right output to the following input:
If it is not appropriate, use flow_input.[input_name]
customer=flow_input.iter.value

Last script: Get orders from shopify
You are in a for loop. You can access one element of the last script's output with the following syntax: flow_input.iter.value
Pipe the right output to the following input:
If it is not appropriate, use flow_input.[input_name]
postgresql_config=flow_input.postgresql_config

Last script: Collect emails and names of accounts
You are in a for loop. You can access one element of the last script's output with the following syntax: flow_input.iter.value
If none are appropriate, use flow_input.[input_name]
account_name=flow_input.iter.value.name
"""


# OUTPUT_TYPE_PROMPT = """
# [INST]
# Script a: {prompt_a}

# Pipe the output of the appropriate script to the following input:
# If none are appropriate, use flow_input.[input_name]
# {input}=
# [/INST]
# """


OUTPUT_TYPE_PROMPT = """
{previous_prompts}
Pipe the output of the appropriate script to the following input:
If none are appropriate, use flow_input.[input_name]
{input}="""

LOOP_OUTPUT_TYPE_PROMPT = """
{previous_prompts}
You are in a for loop. You can access one element of the last script's output with the following syntax: flow_input.iter.value
Pipe the right output to the following input:
If it is not appropriate, use flow_input.[input_name]
{input}="""


def generate_output_type(previous_prompts: str, input: str):
    prompt = EXAMPLES + OUTPUT_TYPE_PROMPT.format(
        previous_prompts=previous_prompts, input=input
    )

    print(prompt)

    response = llm(prompt, max_tokens=20, temperature=0, top_p=1, stop=["\n"])
    return response["choices"][0]["text"]


def generate_loop_output_type(previous_prompts: str, input: str):
    prompt = LOOP_EXAMPLES + LOOP_OUTPUT_TYPE_PROMPT.format(
        previous_prompts=previous_prompts, input=input
    )

    print(prompt)

    response = llm(prompt, max_tokens=20, temperature=0, top_p=1, stop=["\n"])
    return response["choices"][0]["text"]


if __name__ == "__main__":
    # resource_types = """
    # class github(TypedDict):
    #     token: str
    # class slack(TypedDict):
    #     token: str
    # class postgresql(TypedDict):
    #     host: str
    #     port: int
    #     database: str
    #     user: str
    #     password: str
    # """
    # response = generate_script("Get customers from postgresql database", resource_types)
    # print(response)
    # steps = [
    #     "Get customers from postgresql database",
    #     "Loop through customers",
    #     "Print customer name",
    # ]
    # generate_flow([""])
    # example_script = """
    # from typing import Dict, TypedDict
    # class postgresql(TypedDict):
    #     host: str
    #     port: int
    #     database: str
    #     user: str
    #     password: str

    # def main(postgresql_config: Dict[str, postgresql]) -> list:
    #     import psycopg2

    #     config = postgresql_config["postgresql"]
    #     connection = psycopg2.connect(
    #         host=config["host"],
    #         port=config["port"],
    #         dbname=config["database"],
    #         user=config["user"],
    #         password=config["password"]
    #     )

    #     cursor = connection.cursor()
    #     query = "SELECT name FROM customers;"
    #     cursor.execute(query)
    #     customers = cursor.fetchall()
    #     cursor.close()
    #     connection.close()

    #     return customers
    # """
    # previous_prompts = """
    # Script a: Return the name and address from the given company id
    # Script b: Get the account number of orders from the given customer id
    # """
    # inputs = "company_addr"
    # response = generate_output_type(previous_prompts, inputs)
    # print(response)

    previous_prompts = (
        """Last script: return the name and address of companies from the CRM"""
    )
    inputs = "slack"
    response = generate_loop_output_type(previous_prompts, inputs)
    print(response)
