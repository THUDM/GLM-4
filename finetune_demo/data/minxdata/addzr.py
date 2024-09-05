import json
import random

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            if 'image' not in entry['messages'][0]:
                entry['messages'][1]['content'] = 'zRzRzRzRzRzRzR!' + entry['messages'][1]['content']
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

d = read_jsonl('train.jsonl')
save_jsonl(d, 'train.jsonl')
d = read_jsonl('dev.jsonl')
save_jsonl(d, 'dev.jsonl')
