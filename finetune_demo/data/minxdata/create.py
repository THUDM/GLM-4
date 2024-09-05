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
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def select_random_samples(data, num_samples):
    return random.sample(data, num_samples)

# 示例使用
input_train_path = '../AdvertiseGen/train.jsonl'
input_test_path = '../AdvertiseGen/dev.jsonl'

img_train_path = '../small/train.jsonl'
img_test_path = '../small/dev.jsonl'

output_train_path = 'train.jsonl'
output_test_path = 'dev.jsonl'

train_num_samples = 80
test_num_samples = 20

# 读取数据
data1 = read_jsonl(input_train_path)
data2 = read_jsonl(img_train_path)

data3 = read_jsonl(input_test_path)
data4 = read_jsonl(img_test_path)

random_train_samples = select_random_samples(data1, train_num_samples)
random_test_samples = select_random_samples(data3, test_num_samples)

# 保存数据
save_jsonl(random_train_samples + data2, output_train_path)
save_jsonl(random_test_samples + data4, output_test_path)
