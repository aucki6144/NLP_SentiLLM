# -*- coding:utf-8 -*-　
# Last modify: Liu Wentao
# Description: Skeleton for fine-tuning with SemEval data on track A
# Note:


import re
import argparse
import os
import sys

import pandas as pd
from sympy import Integer

sys.path.append(os.path.split(sys.path[0])[0])

from unsloth import FastLanguageModel, get_chat_template
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

prompt_base = """
Categorize the sentence with 5 possible emotions (Anger, Fear, Joy, Sadness, Surprise), represent the mental state of the sentence. 0: no emotion can be inferred. 1: low amount of emotion can be inferred. 2: moderate amount of E can be inferred. 3: hight amount of emotion can be inferred.
Sentence:
"""


def infer(args):
    max_seq_length = args.max_seq_length

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,  # YOUR MODEL YOU USED FOR TRAINING
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

    TP, TN, FP, FN = 0, 0, 0, 0
    distance = 0

    # for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
    for idx, row in df.iterrows():
        sentence = row['text']
        real_emotion = [row['Anger'], row['Fear'], row['Joy'], row['Sadness'], row['Surprise']]

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

        real_output = output_text.group(1).strip()

        # Evaluation
        emotions_intensity = []
        emotions_output = real_output.split(',')

        for emotion_output in emotions_output:
            emotions_intensity.append(int(emotion_output.strip().split("=")[-1]))

        if len(emotions_intensity) != len(real_emotion):
            print(f"Wrong format generated: {real_output}")

        for label, output in zip(real_emotion, emotions_intensity):
            if output == label:
                if output == 0:
                    TN += 1
                else:
                    TP += 1
            else:
                if output == 0:
                    FN += 1
                else:
                    FP += 1
            distance += abs(output - label)

        if args.show_infer:
            print(real_output)

    # Calculate overall metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    overall_precision = TP / (TP + FP)
    overall_recall = TP / (TP + FN)
    overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall)
    avg_distance = distance / (len(df) * 5)

    # Display the results
    print(f"\nOverall Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall: {overall_recall:.4f}")
    print(f"F1 Score: {overall_f1:.4f}")
    print(f"Avg Distance: {avg_distance:.4f}")


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
    parser.add_argument('--show_infer', type=bool, required=False, default=False)

    args = parser.parse_args()
    infer(args)
