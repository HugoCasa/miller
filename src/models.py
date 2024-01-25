import torch
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    T5ForConditionalGeneration,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training


def get_reward_model_and_tokenizer(inference=False, device="auto"):
    model_name = "ise-uiuc/Magicoder-S-DS-6.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # setup lora
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=8,
        lora_dropout=0.1,
        inference_mode=inference,
    )

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        # quantization_config=bnb_config,
        load_in_8bit=True,
        num_labels=1,
        device_map=device,
    )
    # for param in model.model.parameters():
    #     # freeze all parameters except the last layer
    #     param.requires_grad = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    model.config.pad_token_id = model.config.eos_token_id

    # print(f"{model_name} parameters: ")
    # model.print_trainable_parameters()

    return model, tokenizer


def get_model_and_tokenizer(
    model_name, for_ppo=False, for_dpo=False, inference=False, device="auto"
):
    if (
        model_name == "salesforce/codet5p-220m"
        or model_name == "salesforce/codet5p-770m"
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if for_ppo:
            model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_name)
        else:
            model = T5ForConditionalGeneration.from_pretrained(model_name)

        return model, tokenizer
    elif model_name == "ise-uiuc/Magicoder-S-DS-6.7B":
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # setup lora
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=8,
            lora_dropout=0,
            inference_mode=inference,
        )

        if for_ppo:
            model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model_name,
                peft_config=peft_config,
                load_in_8bit=True,
                torch_dtype=torch.bfloat16,
                device_map=device,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
            )
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, peft_config)

            print(f"{model_name} parameters: ")
            model.print_trainable_parameters()

        return model, tokenizer

    else:
        raise ValueError(f"Model {model_name} not supported")
