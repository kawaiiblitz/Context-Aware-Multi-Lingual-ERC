from openai import OpenAI
import pandas as pd
import json
from sklearn.metrics import f1_score, classification_report
import argparse
import os
import time


client = OpenAI(api_key="")

def classify_text(text, context):
    examples = [
        "Classes: [happy, excited, sad, neutral, frustrated, angry]",
        "Text: Oye, que hora es? Esto se acerca a la medianoche, verdad? Dios esto es genial,no?",
        "Class: excited",
        "Text: Mira la noche que tuvimos. No cambiaria esto por nada.",
        "Class: excited",
        "Text: Sabes, en realidad queria ir un poco mas lejos en la costa y alejarme de todas las luces y la gente, pero tenia miedo de que te lo perdieras. Que tal te va?",
        "Class: excited",
        "Text: Es, es, nah, es solo Paul? Ni siquiera puedo decirlo.",
        "Class: excited",
        "---",
        "Text: Asi que te vas manana.",
        "Class: sad",
        "Text: Si. Acaban de llamar.",
        "Class: sad",
        "Text: Esto es realmente injusto.",
        "Class: sad",
        "Text: Lo se.",
        "Class: sad",
        "---",
        "Text: Eso esta fuera de control.",
        "Class: angry",
        "Text: No entiendo por que es tan complicado para la gente cuando llega aqui. Es un simple formulario. Solo necesito una identificacion.",
        "Class: angry",
        "Text: Cuanto tiempo lleva trabajando aqui?",
        "Class: frustrated",
        "---",
        "Text: tu.. No puedo creerlo. Estoy tan feliz por ti. Esto es exactamente lo que querias. Es tu sueno hecho realidad.",
        "Class: excited",
        "Text: Gracias, senor.",
        "Class: happy",
        "Text: El tipo exacto, lo apruebo totalmente. Es un tipo maravilloso. Lo pasamos muy bien. Sabe beber. Eso es genial.",
        "Class: excited",
        "Text: [RISAS]",
        "Class: happy",
        "---",
        "Text: Eso es util.",
        "Class: neutral",
        "Text: Creo que ya no puedo hacer esto, ha pasado mucho tiempo y no es como si no lo intentara.",
        "Class: frustrated",
        "Text: Bueno, supongo que no lo estas intentando lo suficiente.",
        "Class: neutral",
        "Text: Lo intento. Han pasado tres anos.",
        "Class: frustrated"
        "---",
        "Text: Oh. No seas tan grandilocuente. Solo porque resulta que no quieres uno en este momento.",
        "Class: frustrated",
        "Text: No seas estupido.",
        "Class: angry",
        "Text: De verdad, Amanda.",
        "Class: angry",
        "Text: Como?",
        "Class: angry",
        "Text: Nada.",
        "Class: frustrated"
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

    valid_labels = {"happy", "excited", "sad", "neutral", "frustrated", "angry"}
    if label not in valid_labels:
        label = "neutral" 

    #time.sleep(20)

    return label


def process_file(file_path, output_dir, window_size=5):
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