from openai import OpenAI
import pandas as pd
import json
from sklearn.metrics import f1_score, classification_report
import argparse
import os

client = OpenAI(api_key="")

def classify_text(text, context):
    examples = [
        "Classes: [neutral, joy, surprise, anger, sadness, disgust, fear]",
        # Conversation 1
        "Text: Que te pasa?",
        "Class: disgust",
        "Text: Nada!",
        "Class: neutral",
        "Text: Bueno, me dio un dolor cegador en el estomago cuando estaba levantando pesas, luego me desmaye y no he podido levantarme desde entonces.",
        "Class: fear",
        "Text: Pero no creo que sea nada serio.",
        "Class: neutral",
        "Text: Esto suena como una hernia. Tienes que ir al medico!",
        "Class: surprise",
        "Text: No puede ser! Kay mira, si tengo que ir al medico por algo va a ser por esta cosa que me sale del estomago!",
        "Class: fear",
        "---",
        # Conversation 2
        "Text: Me encanta tu casa! De donde es este tipo?",
        "Class: joy",
        "Text: Uh eso es un artefacto indio del siglo XVIII de Calcuta.",
        "Class: neutral",
        "Text: Vaya! Asi que sois mas que dinosaurios.",
        "Class: surprise",
        "Text: Mucho mas.",
        "Class: neutral",
        "---",
        # Conversation 3
        "Text: Ross, puedo hablar contigo un minuto?",
        "Class: neutral",
        "Text: Si, por favor! Entonces, que pasa?",
        "Class: neutral",
        "Text: Uh, bueno... Joey y yo rompimos.",
        "Class: sadness",
        "Text: Dios mio, que ha pasado?",
        "Class: surprise",
        "Text: Joey es un gran tipo, pero somos... tan diferentes! Quiero decir, durante tu discurso no paraba de reirse del homo erectus!",
        "Class: sadness",
        "Text: Sabia que era el!",
        "Class: anger",
        "Text: De todos modos, creo que es lo mejor.",
        "Class: sadness",
         "---",
]    
    examples_text = "\n".join(examples)
    context_text = "\n".join([f"Text: {c['text']}" for c in context])
    
    content = f"""{examples_text}

These are examples of how texts are classified into emotions. Now, given the following context and new text, classify the new text into one of the above classes.

Context:
{context_text}

Text: {text}
Classify the text into one of the above classes. Provide only the class name as the output."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.6,
        messages=[
            {"role": "user", "content": content},
        ]
    )

    label = response.choices[0].message.content.strip()

    valid_labels = {"anger", "surprise", "joy", "sadness", "disgust", "fear", "neutral"}
    if label not in valid_labels:
        label = "neutral" 
    return label


def process_file(file_path, output_dir, window_size=3):
    print(f"Processing file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    true_labels = []
    predicted_labels = []
    
    for dialogue in data:
        for i, utterance in enumerate(dialogue):
            context = dialogue[max(0, i-window_size):i]  
            text = utterance['text']
            true_label = utterance['label']
            predicted_label = classify_text(text, context)
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
    parser.add_argument('--test_file', type=str, required=True, help='Path to the test dataset file.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save the output files.')
    args = parser.parse_args()

    test_true, test_pred = process_file(args.test_file, args.output_dir)

    print("Evaluation on test set:")
    test_f1, test_report = evaluate(test_true, test_pred)

    print("Completed")

if __name__ == "__main__":
    main()