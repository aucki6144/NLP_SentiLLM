# -*- coding:utf-8 -*-　
# Last modify: CHENG Kit Shun
# Description: Roberta Baseline
import argparse
import os
import pandas as pd
import datasets
import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from torch.utils.data import DataLoader, random_split
import numpy as np

def main(args):
    model_name = args.model_name
    data_path = args.data_set

    #Load pretrained model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=5, problem_type="multi_label_classification", ignore_mismatched_sizes=True)

    #Load the dataset
    data_df = pd.read_csv(data_path)

    #convert the labels into numpy array
    def row_process(row):
        text = row['text']
        labels = np.array([row['Anger'], row['Fear'], row['Joy'], row['Sadness'], row['Surprise']])
        labels = labels.astype(np.float32).tolist()
        return {'text': text, 'labels': labels}
    formatted_data = data_df.apply(row_process, axis=1).tolist()

    # Construct datasets
    dataset = datasets.Dataset.from_pandas(pd.DataFrame(formatted_data))
    dataset = dataset.train_test_split(test_size=0.1)
    train_set = dataset['train']
    #train_set = train_set.select(range(0,500))
    test_set = dataset['test']
    #train_set = train_set.select(range(0,100)')

    #preprocess the model_input
    def preprocess(example):
        inputs = "Classify the emotion: " + example['text']
        model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
        input_ids = torch.tensor(model_inputs['input_ids'])
        attention_mask = torch.tensor(model_inputs['attention_mask'])
        labels = torch.tensor(example['labels'])
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    
    train_set = train_set.map(preprocess).remove_columns("text")
    test_set = test_set.map(preprocess).remove_columns("text")
    train_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    print(type(train_set['input_ids']))
    print(len(train_set['labels']))

    train_size = int(0.9* len(train_set))
    val_size = len(train_set) - train_size
    train_dataset, val_dataset = random_split(train_set, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=128)

    #train the model
    optimizer = AdamW(model.parameters(), lr=1e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_function = torch.nn.CrossEntropyLoss()

    for epoch in range(1):    #choose the number of epoch
        model.train()
        for i, batch in enumerate(train_dataloader):
            print(f"{i+1}th batch")
            inputs = {'input_ids': batch['input_ids'],'attention_mask': batch['attention_mask'],'labels': batch['labels']}
            optimizer.zero_grad()
            outputs = model(**inputs)
            logits = outputs.logits
            loss = loss_function(logits, batch['labels'].float())
            loss.backward()
            optimizer.step()



    if args.skip_save:
        print("Skipping saving model")
        return
    else:
        save_dir = os.path.join(args.save_dir, args.model_name.split("/")[-1])
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name', 
        '-m',
        type=str,
        required=False,
        default='cardiffnlp/twitter-roberta-base-emotion',
        help='Path or name to pre-trained model',
    )

    parser.add_argument(
        '--data_set',
        '-d',
        type=str,
        required=False,
        default='./public_data/train/track_a/eng.csv',
        help='Path to fine-tune dataset',
    )

    parser.add_argument(
        '--save_dir',
        '-sd',
        type=str,
        required=False,
        default='./home/checkpoints',
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

    config_args = parser.parse_args()

    main(config_args)
