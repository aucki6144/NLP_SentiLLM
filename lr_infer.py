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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

prompt_base = "Classify the emotion in the sentence:"


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

    true_labels = df[['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']].values.tolist()
    # Prepare to collect predictions
    predicted_outputs = []

    total_samples = len(true_labels)
    true_positive = [0] * 5  # For each emotion: Anger, Fear, Joy, Sadness, Surprise
    false_positive = [0] * 5
    false_negative = [0] * 5
    true_negative = [0] * 5

    emotion_list = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
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
            predicted_emotions = [emotion.strip() for emotion in real_output.split(',') if emotion.strip()]
        else:
            predicted_emotions = []

        predicted_outputs.append(predicted_emotions)

        # Update true positive, false positive, false negative counts
        for i, emotion in enumerate(emotion_list):
            if emotion in predicted_emotions and true_labels[idx][i] == 1:
                true_positive[i] += 1
            elif emotion in predicted_emotions and true_labels[idx][i] == 0:
                false_positive[i] += 1
            elif emotion not in predicted_emotions and true_labels[idx][i] == 1:
                false_negative[i] += 1
            else:
                true_negative[i] += 1

        if args.show_infer:
            print(f"Input: {sentence}\nPredicted Emotions: {predicted_emotions}\n")

    precision_list = []
    recall_list = []
    f1_list = []

    TP = sum(true_positive)
    FP = sum(false_positive)
    FN = sum(false_negative)
    TN = sum(true_negative)

    # Calculate precision, recall, and F1 score for each emotion
    for i, emotion in enumerate(emotion_list):
        precision = true_positive[i] / (true_positive[i] + false_positive[i]) if (true_positive[i] + false_positive[
            i]) > 0 else 0
        recall = true_positive[i] / (true_positive[i] + false_negative[i]) if (true_positive[i] + false_negative[
            i]) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # Calculate overall metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    overall_precision = TP / (TP + FP)
    overall_recall = TP / (TP + FN)
    overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall)

    # Display the results
    print(f"\nOverall Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall: {overall_recall:.4f}")
    print(f"F1 Score: {overall_f1:.4f}")

    # Display metrics for each emotion
    for i, emotion in enumerate(emotion_list):
        print(f"\nMetrics for {emotion}:")
        print(f"  Precision: {precision_list[i]:.4f}")
        print(f"  Recall: {recall_list[i]:.4f}")
        print(f"  F1 Score: {f1_list[i]:.4f}")


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
