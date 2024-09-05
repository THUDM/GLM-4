import json
import random

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for k, v in data.items():
            f.write(json.dumps({k:v}, ensure_ascii=False) + '\n')

# 示例使用
input_train_path = './dev.jsonl'

# 读取数据
data = read_jsonl(input_train_path)

msg = {}

for dd in data:
    if dd['image'] not in msg:
        msg[dd['image']] = []
    msg[dd['image']].append({'question':dd['question'], 'answer':dd['answer']})

save_jsonl(msg, 'mid.jsonl')
