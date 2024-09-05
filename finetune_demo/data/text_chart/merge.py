import json
import random

def read_jsonl(img_path, txt_path):
    with open(img_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        img_data = [json.loads(line) for line in lines]
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        txt_data = [json.loads(line) for line in lines]
    return img_data + txt_data

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

# 示例使用
img_path = './training.jsonl'
txt_path = './text.jsonl'

# 读取数据
data = read_jsonl(img_path, txt_path)
save_jsonl(data, 'final.jsonl')
