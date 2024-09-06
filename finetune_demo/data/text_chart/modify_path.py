import json

def modify(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]

    for entry in data:
        if 'image' in entry['messages'][0]:
            path = '/workspace/ch/LM-Eye-Chart/chart/e_chart/'
            entry['messages'][0]['image'] = '/mnt/ceph/develop/yuxuan/opensource-team/chenhao/GLM-4/finetune_demo/data/text_chart/echart/' + entry['messages'][0]['image'][len(path):]

    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

modify('train.jsonl')
