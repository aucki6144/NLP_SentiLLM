import pandas as pd

import argparse
import os
import torch

import torch.nn.functional as F
from torch.utils.data import Dataset, random_split

from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, get_scheduler,DataCollatorWithPadding

emotions = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']

class MultiEmotionDataset(Dataset):
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

def train_multi_emotion_model(model_name, df, args):
    print(f"Training model for multi emotion")

    # Prepare data
    texts = df['text'].tolist()
    labels = df['labels'].tolist()

    # Load BERT model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = MultiEmotionDataset(texts, labels, tokenizer)

    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    output_dir = str(os.path.join(args.save_dir, model_name.split("/")[-1]))
    print(f"Model will be saved in {output_dir}")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{output_dir}',
        logging_steps=30,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=500,
    )

    optimizer = torch.optim.AdamW(
        [{'params': [param for name, param in model.named_parameters() if 'classifier' not in name]},
         {'params': model.classifier.parameters(), 'lr': 5e-5}], lr=2e-6)
    num_training_steps = len(
        train_dataset) * training_args.num_train_epochs // training_args.per_device_train_batch_size

    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_training_steps
    )

    class MultiEmotionTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop('labels').to('cuda:0').float()
            outputs = model(**inputs)
            logits = outputs.logits
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            return (loss, outputs) if return_outputs else loss

    # Define the Trainer
    trainer = MultiEmotionTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, scheduler)
    )

    # Train the model
    trainer.train()

    if args.skip_save:
        print("Skipping saving model")
        return
    else:
        save_dir = os.path.join(args.save_dir, args.model_name.split("/")[-1])
        print(f"Saving model to {save_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

def main(args):
    model_name = args.model_name
    print(f"Working on {model_name}")
    data_path = args.data_set
    torch.manual_seed(args.seed)
    df = pd.read_csv(data_path)
    df['labels'] = df[emotions].values.tolist()
    train_multi_emotion_model(model_name, df, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        '-m',
        type=str,
        required=False,
        default='bert-base-uncased',
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