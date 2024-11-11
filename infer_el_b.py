# -*- coding:utf-8 -*-
import argparse
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import datasets
from utils import get_model, get_prompt_label_template


def infer(args):
    model_name = args.model_name
    data_path = args.data_set
    torch.manual_seed(args.seed)

    model, tokenizer = get_model(model_name)

    # Load the dataset
    data_df = pd.read_csv(data_path)
    data_df = data_df.sample(frac=0.1)
    prompt_template, label_template = get_prompt_label_template(args.prompt_index)


    # Compose prompts
    def row_process(row):
        raw_text = row['text']
        text = prompt_template.safe_substitute({'sentence': raw_text})
        return {'text': text}

    def row_process_for_result(row):
        text = row['text']
        labels = (f"{row['Anger']} Anger, {row['Fear']} Fear, {row['Joy']} Joy, "
                  f"{row['Sadness']} Sadness, {row['Surprise']} Surprise")

        return {'text': text, 'labels': labels}


    formatted_data = data_df.apply(row_process, axis=1).tolist()
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(formatted_data)

    formatted_testdata = data_df.apply(row_process_for_result, axis=1).tolist()
    test_set = datasets.Dataset.from_pandas(pd.DataFrame(formatted_testdata))

    def seqtonum(a):
        if a == 'zero':
            return '0'
        if a == 'one':
            return '1'
        if a == 'two':
            return '2'
        if a == 'three':
            return '3'
        return '-1'



    ### predicted response to df
    def convert_response2df(label_text):
        elements = label_text.split(", ")
        emotion_data = {'Anger': '0', 'Fear': '0', 'Joy': '0', 'Sadness': '0', 'Surprise': '0'}
        emotion_data['Anger'] = seqtonum(elements[0])
        emotion_data['Fear'] = seqtonum(elements[1])
        emotion_data['Joy'] = seqtonum(elements[2])
        emotion_data['Sadness'] = seqtonum(elements[3])
        emotion_data['Surprise'] = seqtonum(elements[4])
        return emotion_data

    def convert_response2df2(label_text):
        words = label_text.split(", ")
        emotion_data = {'Anger': '-1', 'Fear': '-1', 'Joy': '-1', 'Sadness': '-1', 'Surprise': '-1'}
        elements = []
        for word in words:
            if "=" in word:
                word = word.split('=')[1].strip()
                elements.append(word)
        emotion_data['Anger'] = seqtonum(elements[0])
        emotion_data['Fear'] = seqtonum(elements[1])
        emotion_data['Joy'] = seqtonum(elements[2])
        emotion_data['Sadness'] = seqtonum(elements[3])
        emotion_data['Surprise'] = seqtonum(elements[4])
        return emotion_data

    ### answer to df
    def convert_labeltext2df(label_text):
        elements = label_text.split(", ")
        emotion_data = {}
        for element in elements:
            index, emotion = element.split()
            emotion_data[emotion] = index
        return emotion_data


    # Inferencing
    predict_df = pd.DataFrame(columns=['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise'])
    for example in dataset:
        inputs = example['text']

        # Tokenize the inputs with padding and attention mask
        input_encoding = tokenizer(
            inputs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        input_ids = input_encoding['input_ids']
        attention_mask = input_encoding['attention_mask']

        # Generate prediction with specific EOS token and attention mask
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,  # Pass attention mask to the model
            max_length=256,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,  # Specify EOS token to stop generation
            pad_token_id=tokenizer.pad_token_id,  # Set pad token ID
            no_repeat_ngram_size=2  # Prevent repeating bigrams
        )

        # Decode and print the result
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {inputs}\nPrediction: {prediction}\n")
        emotion_data = convert_response2df2(prediction)
        predict_df = predict_df._append(emotion_data, ignore_index=True)
    #
    # # making true labels df
    true_df = pd.DataFrame(columns=['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise'])
    for index, true_label in enumerate(test_set["labels"]):
        emotion_data = convert_labeltext2df(true_label)
        true_df = true_df._append(emotion_data, ignore_index=True)
    #
    # # confusion_matrix
    emotion_array = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']
    metrics_array = ['TP', 'FP', 'TN', 'FN']
    metrics_df = pd.DataFrame(0, columns=metrics_array, index=emotion_array)
    for i in range(true_df.shape[0]):
        for emotion in emotion_array:
            if true_df.loc[i, emotion] == predict_df.loc[i, emotion] and not true_df.loc[i, emotion] == '0':
                metrics_df.loc[emotion, 'TP'] += 1
            elif true_df.loc[i, emotion] == predict_df.loc[i, emotion] and true_df.loc[i, emotion] == '0':
                metrics_df.loc[emotion, 'TN'] += 1
            elif true_df.loc[i, emotion] != predict_df.loc[i, emotion] and not true_df.loc[i, emotion] == '0':
                metrics_df.loc[emotion, 'FN'] += 1
            elif true_df.loc[i, emotion] != predict_df.loc[i, emotion] and true_df.loc[i, emotion] == '0':
                metrics_df.loc[emotion, 'FP'] += 1
    print(metrics_df)

    # funtion to evalute Accuracy, Precision, Recall and F1 Score (input the df with colume TN, TP, FN and FP)
    def metrics(metrics_df):
        accuracy = (metrics_df['TP'].sum() + metrics_df['TN'].sum()) / metrics_df.sum().sum()

        precision = metrics_df['TP'].sum() / (metrics_df['TP'].sum() + metrics_df['FP'].sum())

        recall = metrics_df['TP'].sum() / (metrics_df['TP'].sum() + metrics_df['FN'].sum())

        F1_score = 2 * (precision * recall) / (precision + recall)
        return {"accuracy score": round(accuracy, 4), "precision": round(precision, 4), "recall": round(recall, 4),
                "F1_score": round(F1_score, 4)}

    # global metrics
    print("global metrics:")
    print(metrics(metrics_df))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        '-m',
        type=str,
        required=False,
        default='./experiments/home/output-c/T5-base',
        help='Path or name to fine-tuned model',
    )

    parser.add_argument(
        '--data_set',
        '-d',
        type=str,
        required=False,
        default='./experiments/home/public_data/train/track_b/eng.csv',
        help='Path to dataset for inference',
    )

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
        default=1,
        help='Index for prompt-label template pair',
    )

    args = parser.parse_args()
    infer(args)
