# -*- coding:utf-8 -*-　
# Last modify: Liu Wentao
# Description:
# Note: THIS CODE CAN ONLY BE RUN ON LINUX

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.split(sys.path[0])[0])

from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj", ]

prompt_base = "Classify the emotion in the sentence:"


def main(args):
    max_seq_length = args.max_seq_length

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    # Get low rank model
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=TARGET_MODULES,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    # Format training data with system prompt template
    def formatting_prompts_func(examples):
        conversions = examples["conversations"]
        texts = [tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) for
                 conv in conversions]
        return {"text": texts, }

    # Process training data
    df = pd.read_csv(args.data_set)
    emotions = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']
    df['labels'] = df[emotions].apply(lambda x: ', '.join([emotions[i] for i, val in enumerate(x) if val == 1]), axis=1)
    df['prompt'] = df['text'].apply(lambda x: f"{prompt_base} '{x}'")

    formatted_data = []
    for _, row in df.iterrows():
        conversation = [
            {'role': 'user', 'content': row['prompt']},
            {'role': 'assistant', 'content': row['labels']}
        ]
        entry = {
            'conversations': conversation,
        }
        formatted_data.append(entry)

    hf_dataset = Dataset.from_list(formatted_data)

    print(f"Training data looks like:\n{hf_dataset[5]}")
    dataset = hf_dataset.map(formatting_prompts_func, batched=True)

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epoch,  # Set this for 1 full training run.
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=args.save_dir,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=training_args,
    )

    # use Unsloth's train_on_completions method to only train on the assistant outputs and ignore the loss on the user's inputs.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    print("Start training")
    trainer_stats = trainer.train()

    model_save_path = os.path.join(args.save_dir, args.model_name.split("/")[-1])
    print(f"Model saved at {model_save_path}")

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        '-m',
        type=str,
        required=False,
        default='unsloth/Llama-3.2-1B-Instruct',
        help='Path or name to pre-trained model',
    )

    parser.add_argument(
        '--data_set',
        '-d',
        type=str,
        required=False,
        default='~/autodl-tmp/data/public_data/train/track_a/eng.csv',
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
        '--epoch',
        '-e',
        type=int,
        required=False,
        default=2,
        help='Epoch number',
    )

    parser.add_argument('--max_seq_length', type=int, required=False, default=512)
    parser.add_argument('--warmup_steps', type=int, required=False, default=5)
    parser.add_argument('--gradient_accumulation_steps', type=int, required=False, default=4)
    parser.add_argument('--learning_rate', type=float, required=False, default=5e-5)
    parser.add_argument('--weight_decay', type=float, required=False, default=0.01)
    parser.add_argument('--logging_steps', type=int, required=False, default=1)

    parser.add_argument('--load_in_4bit', type=bool, required=False, default=True,
                        help='Use 4bit quantization to reduce memory usage. Can be False')

    config_args = parser.parse_args()

    main(config_args)
