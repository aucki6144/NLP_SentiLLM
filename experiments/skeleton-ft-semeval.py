# -*- coding:utf-8 -*-　
# Last modify: Liu Wentao
# Description: Skeleton for fine-tuning with SemEval data on track A
# Note:
import argparse
import pandas as pd
import datasets
import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    DataCollatorWithPadding, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM
import sys
import os

# This line is for adding packages in the root dir into syspath
sys.path.append(os.path.split(sys.path[0])[0])

from utils.prompt_helper import get_prompt_label_template


def main(args):
    model_name = args.model_name
    data_path = args.data_set
    torch.manual_seed(args.seed)

    # Load pretrained model and tokenizer
    if "Llama" in model_name:
        print("Using Llama config")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

        layer_num = 0
        for i, layer in enumerate(model.model.layers):
            print(f"{i} = {layer}")
            layer_num += 1

        for i, layer in enumerate(model.model.layers):
            if i < args.freeze_layer:
                for param in layer.parameters():
                    param.requires_grad = False

        print(
            f"Froze the first {args.freeze_layer} layers, only the last {layer_num - args.freeze_layer} layers will be fine-tuned.")

    elif "T5" in model_name or "t5" in model_name:
        print("Using T5 config")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        print("Unsupported model")
        return

    data_df = pd.read_csv(data_path)

    yes_no_dict = {
        1: 'Yes',
        0: 'No'
    }

    prompt_template, label_template = get_prompt_label_template(args.prompt_index)

    # Compose prompts here
    def row_process(row):
        raw_text = row['text']
        text = prompt_template.safe_substitute({'sentence': raw_text})
        labels = []
        if row['Anger'] == 1:
            labels.append('Anger')
        if row['Fear'] == 1:
            labels.append('Fear')
        if row['Joy'] == 1:
            labels.append('Joy')
        if row['Sadness'] == 1:
            labels.append('Sadness')
        if row['Surprise'] == 1:
            labels.append('Surprise')

        label_str = ", ".join(labels)
        return {'text': text, 'labels': label_str}

    formatted_data = data_df.apply(row_process, axis=1).tolist()

    # Construct sets
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(formatted_data))
    dataset = dataset.train_test_split(test_size=0.1)
    train_set = dataset['train']
    test_set = dataset['test']

    # TODO: Check why batch size error for Llama-3.2-1B
    def preprocess(example):
        inputs = prompt_template.safe_substitute({'sentence': example['text']})

        model_inputs = tokenizer(inputs, padding="max_length", max_length=256, truncation=True)
        labels = tokenizer(example['labels'], padding="max_length", max_length=256, truncation=True)

        model_inputs['labels'] = labels["input_ids"]
        return model_inputs

    train_set = train_set.map(preprocess).remove_columns("text")
    test_set = test_set.map(preprocess).remove_columns("text")

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,  # Due to limited resources
        per_device_eval_batch_size=2,
        weight_decay=0.01,
        save_strategy="no",
        num_train_epochs=2,
        fp16=True,  # Enable mixed precision if using GPU
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define Trainer object
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=test_set,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    if args.skip_save:
        print("Skipping saving model")
        return
    else:
        save_dir = os.path.join(args.save_dir, args.model_name.split("/")[-1])
        print(f"Saving model to {save_dir}")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        '-m',
        type=str,
        required=False,
        default='./home/checkpoints_hf/Llama-3.2-1B-hf',
        help='Path or name to pre-trained model',
    )

    parser.add_argument(
        '--data_set',
        '-d',
        type=str,
        required=False,
        default='./home/data/public_data/train/track_a/eng.csv',
        help='Path to fine-tune dataset',
    )

    parser.add_argument(
        '--save_dir',
        '-sd',
        type=str,
        required=False,
        default='./home/output',
        help='Directory to save fine-tuned checkpoints',
    )

    parser.add_argument(
        '--skip_save',
        '-skip',
        type=bool,
        required=False,
        default=False,
        help='Skip fine-tuning on saved checkpoints',
    )

    parser.add_argument(
        '--batch_size',
        '-b',
        type=int,
        required=False,
        default=4,
        help='Batch size for training',
    )

    # Torch.manual_seed(3407) is all you need
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

    parser.add_argument(
        '--freeze_layer',
        '-f',
        type=int,
        required=False,
        default=0,
        help='Freeze the first n layers',
    )

    config_args = parser.parse_args()

    main(config_args)
