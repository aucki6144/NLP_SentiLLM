# -*- coding:utf-8 -*-　
# Last modify: CHENG Kit Shun
# Description: Skeleton for Evaluation
# Note: WIP

import argparse
import transformers
import datasets
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TextClassificationPipeline
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def evalution(args):

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    data_path = args.data_set

    #data prepossessing
    data_df = pd.read_csv(data_path)
    def row_process(row):
        text = row['text']
        labels = (f"{row['Anger']} Anger, {row['Fear']} Fear, {row['Joy']} Joy, "
                  f"{row['Sadness']} Sadness, {row['Surprise']} Surprise")

        return {'text': text, 'labels': labels}

    formatted_data = data_df.apply(row_process, axis=1).tolist()
    test_set = datasets.Dataset.from_pandas(pd.DataFrame(formatted_data))
    #test_set = test_set.select(range(0,50))

    #function to make prediction
    def predict(input_text, model, tokenizer):
        inputs = "Classify the emotion: " + input_text
        model_inputs = tokenizer(inputs, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        outputs = model.generate(**model_inputs, max_new_tokens=100)
        predicted_labels = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(predicted_labels)
        return predicted_labels
    
    #convert the label in text format to dictionary format
    def convert_labeltext2df(label_text):
        elements = label_text.split(", ")
        emotion_data = {}
        for element in elements:
            index, emotion = element.split()
            emotion_data[emotion] = index
        return emotion_data

    model.eval()

    #making prediction df
    predict_df = pd.DataFrame(columns=['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise'])
    for index, text in enumerate(test_set["text"]):
        label = predict(text, model, tokenizer)
        emotion_data = convert_labeltext2df(label)
        predict_df = predict_df._append(emotion_data, ignore_index=True)

    #making true labels df
    true_df = pd.DataFrame(columns=['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise'])
    for index, true_label in enumerate(test_set["labels"]):
        emotion_data = convert_labeltext2df(true_label)
        true_df = true_df._append(emotion_data, ignore_index=True)


    #confusion_matrix
    emotion_array = ['Anger', 'Fear', 'Joy', 'Sadness', 'Surprise']
    metrics_array = ['TP', 'FP', 'TN', 'FN']
    metrics_df = pd.DataFrame(0, columns=metrics_array, index = emotion_array)
    for i in range(true_df.shape[0]):
        for emotion in emotion_array:
            if true_df.loc[i, emotion] == predict_df.loc[i, emotion] and true_df.loc[i, emotion]== '1':
                metrics_df.loc[emotion, 'TP'] += 1
            elif true_df.loc[i, emotion] == predict_df.loc[i, emotion] and true_df.loc[i, emotion] == '0':
                metrics_df.loc[emotion, 'TN'] += 1
            elif true_df.loc[i, emotion] != predict_df.loc[i, emotion] and true_df.loc[i, emotion] == '1':
                metrics_df.loc[emotion, 'FN'] += 1
            elif true_df.loc[i, emotion] != predict_df.loc[i, emotion] and true_df.loc[i, emotion] == '0':
                metrics_df.loc[emotion, 'FP'] += 1
    print(metrics_df)        

    #funtion to evalute Accuracy, Precision, Recall and F1 Score (input the df with colume TN, TP, FN and FP)
    def metrics(metrics_df):
        accuracy = (metrics_df['TP'].sum()+metrics_df['TN'].sum())/metrics_df.sum().sum()

        precision = metrics_df['TP'].sum()/(metrics_df['TP'].sum()+metrics_df['FP'].sum())

        recall = metrics_df['TP'].sum()/(metrics_df['TP'].sum()+metrics_df['FN'].sum())

        F1_score = 2*(precision*recall)/(precision+recall)
        return {"accuracy score": round(accuracy, 4), "precision": round(precision,4), "recall": round(recall,4), "F1_score": round(F1_score,4)}

    #global metrics
    print("global metrics:")
    print(metrics(metrics_df))

    #metrics of every emotion
    for emotion in emotion_array:
        print(emotion)
        print(metrics(metrics_df.loc[emotion]))
    return







if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path',
        '-m',
        type=str,
        required=False,
        default='./home/checkpoints/t5-small',
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