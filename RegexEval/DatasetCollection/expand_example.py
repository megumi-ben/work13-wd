import json
import random
import re

# 输入输出文件路径
input_file_path = "/home/lyz/wd/LLM_Regular/RegexEval/DatasetCollection/regex_pos_pairs.jsonl"
output_file_path = "/home/lyz/wd/LLM_Regular/RegexEval/DatasetCollection/expanded_regex_pos_pairs.jsonl"

# 目标样本数量
target_samples = 5000000
max_samples_per_regex = 10000  # 每个正则生成的最大样本数

def expand_samples_for_regex(regex, current_samples, num_samples=10000):
    """
    使用正则引擎生成更多正样例。
    返回一个列表，包含生成的正样本。
    """
    expanded_samples = set(current_samples)
    
    # 根据正则生成新的正样例
    for _ in range(num_samples):
        if regex == "^\\d$":
            new_sample = str(random.randint(0, 9))  # 随机生成 0-9 的数字
            expanded_samples.add(new_sample)
        elif regex == "^\\d{5}$":
            new_sample = f"{random.randint(10000, 99999)}"  # 随机生成 5 位数字
            expanded_samples.add(new_sample)
        # 可以根据需要继续添加其他正则的扩展规则
    
    return list(expanded_samples)

# 读取现有数据
with open(input_file_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# 计算总共需要生成多少样本
current_samples_count = len(data)
print(f"当前样本数: {current_samples_count}")

# 用于存储生成的扩展数据
expanded_data = []
current_samples = 0

# 对每个正则表达式进行扩展
for entry in data:
    regex = entry['regex']
    matches = entry['positive_example']
    
    # 扩展每个正则的正样本
    expanded_samples = expand_samples_for_regex(regex, [matches], num_samples=max_samples_per_regex)
    
    # 将扩展后的样本添加到结果中
    for sample in expanded_samples:
        expanded_data.append({
            "regex": regex,
            "positive_example": sample
        })
        
    current_samples += len(expanded_samples)
    if current_samples >= target_samples:
        break

# 写入到新的 JSONL 文件中
with open(output_file_path, "w", encoding="utf-8") as f:
    for entry in expanded_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"生成的数据集保存到: {output_file_path}")
print(f"总样本数: {len(expanded_data)}")
