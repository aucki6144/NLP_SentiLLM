# -*- coding:utf-8 -*-　
# Last modify: Liu Wentao
# Description: Skeleton for fine-tuning with SemEval data on track A
# Note:
import argparse

import torch
from transformers import pipeline

def infer(args):
    model_name = args.model_name

    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(pipe(args.content))

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
        '--content',
        '-c',
        type=str,
        required=True,
        help='Prompt for inference',
    )

    args = parser.parse_args()
    infer(args)
