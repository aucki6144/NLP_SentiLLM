# -*- coding:utf-8 -*-　
# Last modify: Liu Wentao
# Description: Baseline BERT, use 5 models, Score compute
# Note:

import argparse
import os
import sys

# This line is for adding packages in the root dir into syspath
sys.path.append(os.path.split(sys.path[0])[0])

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, RobertaForSequenceClassification, \
    RobertaTokenizer
from torch.utils.data import Dataset, DataLoader

emotions = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

# Initialize containers for total metrics
all_labels = []
all_preds = []


# Define a custom dataset class
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Function to evaluate a single emotion model
def evaluate_emotion_model(device, model_name, df, emotion, args):
    print(f"Evaluating model for {emotion}...")

    # Load the trained model
    model = RobertaForSequenceClassification.from_pretrained(os.path.join(model_name, emotion))
    model.to(device)
    tokenizer = RobertaTokenizer.from_pretrained(os.path.join(model_name, emotion))
    model.eval()

    # Prepare data
    texts = df['text'].tolist()
    labels = df[emotion].tolist()
    eval_dataset = EmotionDataset(texts, labels, tokenizer)

    # Initialize lists to collect predictions and labels
    preds = []
    labels = []

    # Use DataLoader for batch inference
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size)

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Inference for {emotion}", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            label = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1)

            preds.extend(pred.cpu().numpy())
            labels.extend(label.cpu().numpy())

    # Calculate metrics for the current emotion
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    print(f"{emotion} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    result = f"{emotion} - Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n"

    return result, labels, preds


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    final_result = ""
    model_name = args.model_name
    print(f"Working on {model_name}")
    data_path = args.data_set
    torch.manual_seed(args.seed)
    df = pd.read_csv(data_path)

    for emotion in emotions:
        # Prepare evaluation dataset for the current emotion

        # Evaluate and collect metrics
        re, labels, preds = evaluate_emotion_model(device, model_name, df, emotion, args)
        final_result += re
        # Append results to total lists
        all_labels.extend(labels)
        all_preds.extend(preds)

    # Calculate overall metrics
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(all_labels, all_preds,
                                                                                       average='binary')
    overall_accuracy = accuracy_score(all_labels, all_preds)

    print(final_result, end='')
    print("\nOverall Metrics:")
    print(
        f"Accuracy: {overall_accuracy:.4f}, Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1: {overall_f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        '-m',
        type=str,
        required=False,
        default='FacebookAI/roberta-base',
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
        default=1,
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
