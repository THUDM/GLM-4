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
            for k, v in entry.items():
                msg = []
                for i in range(len(v)):
                    mku = {}
                    mka = {}
                    if i == 0:
                        mku['image'] = k
                    mku['role'] = 'user'
                    mku['content'] = v[i]['question']
                    mka['role'] = 'assistant'
                    mka['content'] = v[i]['answer']
                    msg.append(mku)
                    msg.append(mka)

            f.write(json.dumps({'messages':msg}, ensure_ascii=False) + '\n')

# 示例使用
input_train_path = './mid.jsonl'

# 读取数据
data = read_jsonl(input_train_path)

save_jsonl(data, 'deving.jsonl')
