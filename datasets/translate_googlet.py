import json
from googletrans import Translator
import time

translator = Translator()

def translate_text(text):
    try:
        translated = translator.translate(text, src='en', dest='es')
        return translated.text
    except Exception as e:
        print(f"Error translating text: {text}\nError: {e}")
        return text 

def translate_json_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as file:
        dialogues = json.load(file)

    for dialogue in dialogues:
        for utterance in dialogue:
            if 'text' in utterance and utterance['text'].strip():
                utterance['text'] = translate_text(utterance['text'])
                #time.sleep(1)  

    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(dialogues, file, ensure_ascii=False, indent=4)

#train_path_iemocap = '/home/raquel/dev/codebase/ERCMC/datasets/final/meld/part2.json'
#test_path_iemocap = '/home/raquel/dev/codebase/ERCMC/datasets/final/meld/test.json'
dev_path_iemocap = '/home/raquel/dev/codebase/ERCMC/datasets/final/meld/dev.json'

#train_output_path_iemocap = '/home/raquel/dev/codebase/ERCMC/datasets/final_spa_googlet/meld/train_part2.json'
#test_output_path_iemocap = '/home/raquel/dev/codebase/ERCMC/datasets/final_spa_googlet/meld/test.json'
dev_output_path_iemocap = '/home/raquel/dev/codebase/ERCMC/datasets/final_spa_googlet/meld/dev.json'

#translate_json_file(train_path_iemocap, train_output_path_iemocap)
#translate_json_file(test_path_iemocap, test_output_path_iemocap)
translate_json_file(dev_path_iemocap, dev_output_path_iemocap)

