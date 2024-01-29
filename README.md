# Automated Script and Workflow Generation using Large Language Models

This is the official repository for my Master's thesis, titled "Automated Script and Workflow Generation using Large Language Models". It contains the code used to train the models for generating, fixing and editing scripts in Python and Deno for the [Windmill platform](https://windmill.dev). It also includes the system for workflow generation.

All models' weights are available on my [HuggingFace profile](https://huggingface.co/HugoCasa).

Here is an overview of the files in this repository:
- `src/main.py`, `src/trainer.py`, `src/dpo.py`, `src/prompts.py`, `src/models.py` and `src/datasets.py` contain the code for training the models. It is setup to run using DDP.
- `src/eval.py`, `src/benchmark.py`, and `src/inference.py` contain the code for evaluating the models.
- `src/resource_types.py` and `src/openapi.py` contain the code for the pre-processing and retrieval systems of the resource types and OpenAPI information.
- `src/synthetic_*` files contain the code for collecting the prompts and generating the synthetic data for gen, edit and fix tasks.
- `src/mbpp_processing.py` and `src/humaneval_processing.py` contain the code for adapting the MBPP and Humaneval datasets
- `src/dpo_gen.py` and `src/dpo_pairs_scoring.py` contain the code for the additional DPO generation and ranking.
- `src/data_prep.py` contains the data for compiling the complete dataset for training (generation task only).
- `src/convertor.py` contains the code to convert the data between the different programming languages.
- `ìnference_speed.py` and `inference_speed_apis.py` contain the code for measuring the inference speed of the models.
- `eval_embeddings.py` and `eval_twostepgen.py` contain the code for evaluating the embeddings and the two-step synthetic data generation.

You need to create an `.env` file in the root folder with an OPENAI_API_KEY variable to run the data generation files.

The data used for training and inference (resource types + OpenAPI embeddings) is in the `data` folder. The benchmark data is not included as it contains credentials.
The validation and benchmark results can be found in the `models` folder.
The experiment logs are available on [Weights & Biases](https://wandb.ai/hugocasa/projects).

You can easily demo the models using the `demo.py` script. The script will download the model weights from HuggingFace and run the demo.
You can call it like this:
  
```bash
python demo.py --lang "python" --model_name="hugocasa/miller-6.7B-openapi-aligned" --kind "gen" --instructions "list the commits of a github repository"
```

The linking for the workflow generation can be tested in the `flow_generation.py` script. It uses the Llama.cpp implementation and requires downloading the [5-bit quantized GGUF weights of Mistral 7B v0.1](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/blob/main/mistral-7b-v0.1.Q5_K_M.gguf) into the `models/mistral` folder.