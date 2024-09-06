import requests
import json
import time

API_KEY = '414d6693-1c43-4291-820b-c429dd9e629e'
API_URL = 'https://api.deepl.com/v2/translate'

def translate_text(text, source_lang='EN', target_lang='ES'):
    params = {
        'auth_key': API_KEY,
        'text': text,
        'source_lang': source_lang,
        'target_lang': target_lang
    }

    response = requests.post(API_URL, data=params)
    if response.status_code == 200:
        result = response.json()
        return result['translations'][0]['text']
    else:
        return f"Error: {response.status_code}, {response.text}"

def read_input_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def write_output_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def main(input_file, output_file):
    data = read_input_file(input_file)
    
    for dialogue in data:
        for sentence in dialogue:
            original_text = sentence["text"]
            translated_text = translate_text(original_text)
            sentence["text"] = translated_text
            print(f"Original: {original_text} -> Translated: {translated_text}")
            time.sleep(1)  # Pause of 1 second
    
    write_output_file(output_file, data)
    print(f"Translation completed and saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Translate text using DeepL API')
    parser.add_argument('input_file', type=str, help='Path to the input JSON file')
    parser.add_argument('output_file', type=str, help='Path to the output JSON file')
    
    args = parser.parse_args()
    
    main(args.input_file, args.output_file)
