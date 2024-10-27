# -*- coding:utf-8 -*-
import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

from utils.prompt_helper import get_prompt_label_template


def infer(args):
    model_name = args.model_name
    data_path = args.data_set
    torch.manual_seed(args.seed)

    # Load the trained model and tokenizer
    print("Loading the trained model...")
    if "Llama" in model_name:
        print("Using Llama config")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    elif "T5" in model_name or "t5" in model_name:
        print("Using T5 config")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        print("Unsupported model")
        return

    # Ensure padding token is set
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    # Load the dataset
    data_df = pd.read_csv(data_path)

    prompt_template, label_template = get_prompt_label_template(args.prompt_index)

    # Compose prompts
    def row_process(row):
        raw_text = row['text']
        text = prompt_template.safe_substitute({'sentence': raw_text})
        return {'text': text}

    formatted_data = data_df.apply(row_process, axis=1).tolist()

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(formatted_data)

    # Inferencing
    for example in dataset:
        inputs = example['text']

        # Tokenize the inputs with padding and attention mask
        input_encoding = tokenizer(
            inputs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        ).to(model.device)

        input_ids = input_encoding['input_ids']
        attention_mask = input_encoding['attention_mask']

        # Generate prediction with specific EOS token and attention mask
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,  # Pass attention mask to the model
            max_length=256,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,  # Specify EOS token to stop generation
            pad_token_id=tokenizer.pad_token_id,  # Set pad token ID
            no_repeat_ngram_size=2  # Prevent repeating bigrams
        )

        # Decode and print the result
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {inputs}\nPrediction: {prediction}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        '-m',
        type=str,
        required=True,
        help='Path or name to fine-tuned model',
    )

    parser.add_argument(
        '--data_set',
        '-d',
        type=str,
        required=True,
        help='Path to dataset for inference',
    )

    parser.add_argument(
        '--seed',
        '-s',
        type=int,
        required=False,
        default=3407,
        help='Random seed for reproducibility',
    )

    parser.add_argument(
        '--prompt_index',
        '-ind',
        type=int,
        required=False,
        default=0,
        help='Index for prompt-label template pair',
    )

    args = parser.parse_args()
    infer(args)
