from openai import OpenAI
import pandas as pd
import json
from sklearn.metrics import f1_score, classification_report
import argparse
import os

client = OpenAI(api_key=")

def classify_text(text):
    content = f"""Classes: [happy, excited, sad, neutral, frustrated, angry]
Text: {text}

Classify the text into one of the above classes. Provide only the class name as the output."""

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.6,
        messages=[
            {"role": "user", "content": content},
        ]
    )

    label = response.choices[0].message.content.strip()
    return label


def process_file(file_path, output_dir):
    print(f"Processing file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    true_labels = []
    predicted_labels = []
    
    for dialogue in data:
        for utterance in dialogue:
            text = utterance['text']
            true_label = utterance['label']
            predicted_label = classify_text(text)
            utterance['predicted_label'] = predicted_label
            
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
    
    base_name = os.path.basename(file_path)
    output_file = os.path.join(output_dir, base_name.replace(".json", "_with_predictions.json"))
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Saved: {output_file}")
    
    return true_labels, predicted_labels

def evaluate(true_labels, predicted_labels):
    weighted_f1 = f1_score(true_labels, predicted_labels, average='weighted')
    report = classification_report(true_labels, predicted_labels)
    print(f"Weighted F1-score: {weighted_f1}")
    print(f"Classification Report:\n{report}")
    return weighted_f1, report

def main():
    parser = argparse.ArgumentParser(description="Process and evaluate text classification using OpenAI GPT-4.")
    parser.add_argument('--train_file', type=str, required=True, help='Path to the training dataset file.')
    parser.add_argument('--test_file', type=str, required=True, help='Path to the test dataset file.')
    parser.add_argument('--dev_file', type=str, required=True, help='Path to the dev dataset file.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save the output files.')
    args = parser.parse_args()

    train_true, train_pred = process_file(args.train_file, args.output_dir)
    test_true, test_pred = process_file(args.test_file, args.output_dir)
    dev_true, dev_pred = process_file(args.dev_file, args.output_dir)

    print("Evaluation on training set:")
    train_f1, train_report = evaluate(train_true, train_pred)

    print("Evaluation on test set:")
    test_f1, test_report = evaluate(test_true, test_pred)

    print("Evaluation on dev set:")
    dev_f1, dev_report = evaluate(dev_true, dev_pred)

    print("Completed")

if __name__ == "__main__":
    main()