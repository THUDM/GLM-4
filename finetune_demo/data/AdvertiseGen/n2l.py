import json

# 读取原始 JSON 文件
with open('train.json', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 将每行 dict 转换为 JSONL 格式并写入新文件
with open('train.jsonl', 'w', encoding='utf-8') as f_out:
    for line in lines:
        # 去除行尾的换行符并解析为字典
        dict_data = json.loads(line.strip())
        dict_data = {'messages': [{'role': 'user', 'content': dict_data['content']}, {'role': 'assistant', 'content': dict_data['summary']}]}
        # 将字典转换为 JSON 字符串
        json_line = json.dumps(dict_data, ensure_ascii=False)
        # 写入文件并换行
        f_out.write(json_line + '\n')

