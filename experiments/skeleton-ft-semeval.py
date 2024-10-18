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

from utils.prompt_helper import get_prompt_template


def main(args):
    model_name = args.model_name
    data_path = args.data_set
    torch.manual_seed(args.seed)

    # Load pretrained model and tokenizer
    if "Llama" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    elif "T5" in model_name:
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

    def row_process(row):
        text = row['text']
        labels = (f"Angry: {yes_no_dict[row['Anger']]}, "
                  f"Fear: {yes_no_dict[row['Fear']]}, "
                  f"Joy: {yes_no_dict[row['Joy']]}, "
                  f"Sadness: {yes_no_dict[row['Sadness']]}, "
                  f"Surprise:{yes_no_dict[row['Surprise']]}")
        # print(labels)

        return {'text': text, 'labels': labels}

    formatted_data = data_df.apply(row_process, axis=1).tolist()

    # Construct sets
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(formatted_data))
    dataset = dataset.train_test_split(test_size=0.1)
    train_set = dataset['train']
    test_set = dataset['test']

    prompt_template = get_prompt_template()

    def preprocess(example):
        inputs = prompt_template.safe_substitute({'sentence': example['text']})
        # print(inputs)

        model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(example['labels'], max_length=32, padding="max_length", truncation=True)

        model_inputs['labels'] = labels["input_ids"]
        return model_inputs

    train_set = train_set.map(preprocess).remove_columns("text")
    test_set = test_set.map(preprocess).remove_columns("text")

    # print(train_set[0])
    # print(test_set[0])

    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=1,  # Due to limited resources
        per_device_eval_batch_size=1,
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
        default='./home/checkpoints_hf/T5-small-hf',
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

    config_args = parser.parse_args()

    main(config_args)
