# -*- coding:utf-8 -*-　
# Last modify: Liu Wentao
# Description: Skeleton for fine-tuning with SemEval data on track A
# Note:


import re
import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.split(sys.path[0])[0])

from unsloth import FastLanguageModel, get_chat_template

prompt_base = "Classify the emotion in the sentence:"


def infer(args):
    max_seq_length = args.max_seq_length

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model",  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    # Load data
    df = pd.read_csv(args.data_set)

    for idx, row in df.iterrows():
        sentence = row['text']

        # Prepare messages for input
        messages = [
            {"role": "user", "content": f"{prompt_base} {sentence}"},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=64,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
        )

        inference_output = tokenizer.batch_decode(outputs)
        output_text = re.search(
            r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>',
            inference_output[0],
            re.DOTALL,
        )

        if output_text:
            real_output = output_text.group(1).strip()
            print(f"Input: {sentence}\nPredicted Emotions: {real_output}\n")
        else:
            print(f"Input: {sentence}\nNo assistant response found.\n")


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
        help='Dataset path',
    )

    parser.add_argument(
        '--content',
        '-c',
        type=str,
        required=False,
        help='Prompt for inference',
    )

    parser.add_argument('--max_seq_length', type=int, required=False, default=512)
    parser.add_argument('--load_in_4bit', type=bool, required=False, default=True,
                        help='Use 4bit quantization to reduce memory usage. Can be False')

    args = parser.parse_args()
    infer(args)
