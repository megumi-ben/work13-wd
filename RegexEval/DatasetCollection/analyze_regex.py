import json

# 输入文件路径
input_file_path = "/home/lyz/wd/LLM_Regular/RegexEval/DatasetCollection/regex_pos_pairs.jsonl"

# 用来存储所有正则表达式模板
regex_set = set()

# 读取数据并提取正则模板
with open(input_file_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

    for entry in data:
        regex = entry['regex']
        regex_set.add(regex)

# 打印所有正则表达式模板
print(f"共有 {len(regex_set)} 个不同的正则表达式模板：")
for regex in regex_set:
    print(regex)
