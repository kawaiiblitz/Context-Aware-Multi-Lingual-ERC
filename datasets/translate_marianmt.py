import json
from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-en-es'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate_text(text):
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_json_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        dialogues = json.load(file)

    for dialogue in dialogues:
        for utterance in dialogue:
            utterance['text'] = translate_text(utterance['text'])

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(dialogues, file, ensure_ascii=False, indent=4)

train_path = '/home/raquel/dev/codebase/ERCMC/datasets/final/meld/train.json'
test_path = '/home/raquel/dev/codebase/ERCMC/datasets/final/meld/test.json'
dev_path = '/home/raquel/dev/codebase/ERCMC/datasets/final/meld/dev.json'

train_output_path = '/home/raquel/dev/codebase/ERCMC/datasets/final_spa_marianmt/meld/train.json'
test_output_path = '/home/raquel/dev/codebase/ERCMC/datasets/final_spa_marianmt/meld/test.json'
dev_output_path = '/home/raquel/dev/codebase/ERCMC/datasets/final_spa_marianmt/meld/dev.json'

translate_json_file(train_path, train_output_path)
translate_json_file(test_path, test_output_path)
translate_json_file(dev_path, dev_output_path)

