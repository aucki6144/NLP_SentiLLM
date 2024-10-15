import argparse
import os

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    DataCollatorWithPadding, DataCollatorForSeq2Seq


def main(args):
    print(f"Executing fine-tuning on {args.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # TODO: Transfer to dataset by SemEVAL2025
    dataset = load_dataset(args.data_set)
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    print(train_dataset.features)

    id2label = {idx: label for idx, label in enumerate(train_dataset.features["label"].names)}
    label2id = {label: idx for idx, label in id2label.items()}

    def preprocess_function(example):
        inputs = "Classify the emotion: " + example['text']
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)

        labels = id2label[example['label']]
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(labels, max_length=32, truncation=True)

        model_inputs['labels'] = labels["input_ids"]
        return model_inputs

    train_dataset = train_dataset.map(preprocess_function)
    val_dataset = val_dataset.map(preprocess_function)

    train_dataset = train_dataset.remove_columns("text").remove_columns("label")
    val_dataset = val_dataset.remove_columns("text").remove_columns("label")

    print(train_dataset[0])
    print(val_dataset[0])

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,  # Due to limited resources
        per_device_eval_batch_size=2,
        weight_decay=0.01,
        save_strategy="no",
        num_train_epochs=1,
        fp16=True,  # Enable mixed precision if using GPU
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define Trainer object
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # TODO: Construct evaluate pipeline
    # Evaluate the model on the validation set
    # results = trainer.evaluate()
    # print(f"Validation Loss: {results['eval_loss']}")

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
        default='google-t5/t5-small',
        help='Path or name to pre-trained model',
    )

    parser.add_argument(
        '--data_set',
        '-d',
        type=str,
        required=False,
        default='emotion',
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
        default=True,
        help='Skip fine-tuning on saved checkpoints',
    )

    config_args = parser.parse_args()

    main(config_args)