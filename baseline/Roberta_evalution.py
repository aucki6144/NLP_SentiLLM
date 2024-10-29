# -*- coding:utf-8 -*-　
# Last modify: CHENG Kit Shun
# Description: Evalution of Roberta model

import argparse
import transformers
import datasets
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from sklearn.metrics import precision_recall_fscore_support, hamming_loss
from torch.utils.data import DataLoader
import numpy as np
def evalution(args):
    #load the model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
    model = RobertaForSequenceClassification.from_pretrained(args.model_path)
    data_path = args.data_set

    #data prepossessing
    data_df = pd.read_csv(data_path)
    def row_process(row):
        text = row['text']
        labels = np.array([row['Anger'], row['Fear'], row['Joy'], row['Sadness'], row['Surprise']])
        labels = labels.astype(np.float32).tolist()
        return {'text': text, 'labels': labels}
    

    formatted_data = data_df.apply(row_process, axis=1).tolist()
    test_set = datasets.Dataset.from_pandas(pd.DataFrame(formatted_data))
    #test_set = test_set.select(range(0,500))

    def preprocess(example):
        inputs = "Classify the emotion: " + example['text']
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
        input_ids = torch.tensor(model_inputs['input_ids'])
        attention_mask = torch.tensor(model_inputs['attention_mask'])
        labels = np.array(example['labels']).astype(np.float32)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    test_set = test_set.map(preprocess).remove_columns("text")
    y_true = test_set['labels']
    test_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataloader = DataLoader(test_set, batch_size=128)

    #Prediction
    model.eval()
    y_pred = []
    threshold = 0.5
    for i, batch in enumerate(test_dataloader):
        with torch.no_grad():
            print(f"{i+1}th batch")
            inputs = {'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask']}
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_labels = (torch.sigmoid(logits) > threshold).float().numpy()
            print(predicted_labels)
            print(batch['labels'])
            y_pred.extend(predicted_labels) 

    #metrics evalution
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=1)
    accuracy = 1 - hamming_loss(y_true, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    return







if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path',
        '-m',
        type=str,
        required=False,
        default='./home/checkpoints/twitter-roberta-base-emotion',
        help='Path to fine-tuned model',
    )

    parser.add_argument(
        '--data_set',
        '-d',
        type=str,
        required=False,
        default='./public_data/train/track_a/eng.csv',
        help='Path to evaluation dataset',
    )


    config_args = parser.parse_args()

    evalution(config_args)
