# Miller models

This is the official repository for my Master's thesis, titled "Automated Script and Workflow Generation using Large Language Models". It contains the code used to train the models for generating, fixing and editing scripts in Python and Deno for the [Windmill platform](https://windmill.dev). It also contains the system for workflow generation.

All models' weights are available on my [HuggingFace profile](https://huggingface.co/HugoCasa).

The files inside the `src` folder contain the data processing and training code. The training is set up to run using DDP.
All the data used for training and inference (resource types + OpenAPI embeddings) is available in the `data` folder. The benchmark data is not included as it contains credentials.
The validation and benchmark results can be found in the `models` folder.
The experiment logs are available on [Weights & Biases](https://wandb.ai/hugocasa/projects).

You can easily demo the models using the `demo.py` script. The script will download the model weights from HuggingFace and run the demo.
You can call it like this:
  
```bash
python demo.py --lang "python" --model_name="hugocasa/miller-6.7B-openapi-aligned" --kind "gen" --instructions "list the commits of a github repository"
```

The linking for the workflow generation can be tested in the `flow_generation.py` script. It uses the Llama.cpp implementation and requires downloading the [5-bit quantized GGUF weights of Mistral 7B v0.1](https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/blob/main/mistral-7b-v0.1.Q5_K_M.gguf) into the `models/mistral` folder.