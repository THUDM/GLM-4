import json
import random

def select_random_samples(input_file, output_file, num_samples=3000):
    # 读取所有行
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    # 确保样本数量不超过总行数
    num_samples = min(num_samples, len(lines))
    
    # 随机选择5000条
    selected_lines = random.sample(lines, num_samples)
    
    # 将选中的数据保存到新的 JSONL 文件中
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.writelines(selected_lines)

# 使用示例
input_file = 'train.jsonl'
output_file = 'xl.jsonl'
select_random_samples(input_file, output_file)

