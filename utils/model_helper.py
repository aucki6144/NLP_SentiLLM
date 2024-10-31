# -*- coding:utf-8 -*-　
# Last modify: Liu Wentao
# Description: Skeleton for fine-tuning with SemEval data on track A
# Note:
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering


def get_model(model_name):
    # Load the trained model and tokenizer
    print("Loading the trained model...")
    if "Llama" in model_name:
        print("Using Llama config")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
        model.resize_token_embeddings(len(tokenizer))
    elif "T5" in model_name or "t5" in model_name:
        print("Using T5 config")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        print("Unsupported model")
        return

    return model, tokenizer


def get_model_train(model_name, freeze_layer = 0):
    if "Llama" in model_name:
        print("Using Llama config")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
        model.resize_token_embeddings(len(tokenizer))

        layer_num = 0
        for i, layer in enumerate(model.model.layers):
            print(f"{i} = {layer}")
            layer_num += 1

        for i, layer in enumerate(model.model.layers):
            if i < freeze_layer:
                for param in layer.parameters():
                    param.requires_grad = False

        print(
            f"Froze the first {freeze_layer} layers, only the last {layer_num - freeze_layer} layers will be fine-tuned.")

    elif "T5" in model_name or "t5" in model_name:
        print("Using T5 config")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        print("Unsupported model")
        return

    return model, tokenizer