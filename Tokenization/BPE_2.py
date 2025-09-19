#!/usr/bin/python
# -*- coding:utf-8 -*-
import itertools
from collections import defaultdict

def BPETokenization(input, max_merge_time=10):
    # 1. 先把input包含的所有字符转化为对应的字节
    indices = []
    bytes_input = input.encode('utf-8')
    for byte in bytes_input:
        int_value = int(byte)
        indices.append(int_value)

    # print(indices)

    # 2. 创建合并规则表：{(int, int), int)} -》 token对，对应的序列号
    merges = {}

    # 3. 初始化词表：初始为全部的单字节，方便后续的合并出新的更高效的token
    vocab = {x: bytes([x]) for x in range(256)}

    # 2. 循环执行 BPE 合并
    for i in range(max_merge_time):
        # 2.1 记录每个token对出现的次数
        counts = defaultdict(int)
        for (index1, index2) in zip(indices, indices[1:]):
            counts[(index1, index2)] += 1
        if not counts:
            break
        # 2.2 选出token对出现次数最多的一对
        pairs = max(counts, key=counts.get)
        index1, index2 = pairs

        # 2.3 为新 token 分配一个 id（从 256 开始）
        new_index = 256 + i

        # 2.4 新增合并规则
        merges[pairs] = new_index
        # 2.5 更新词表，新token的字符串=两个旧token的拼接
        vocab[new_index] = vocab[index1] + vocab[index2]
        print(f"合并 {vocab[index1]} + {vocab[index2]} -> {vocab[new_index]}")


        # 2.6 在文本中执行合并：把 (index1, index2) 替换成 new_index
        indices = merge(indices, pairs, new_index)

        print("当前序列:", [vocab[idx] for idx in indices])

    return vocab, merges

def merge(indices, pair, new_index):
    """
    把序列中所有 pair 替换为 new_index
    """
    output = []
    skip = False
    for i in range(len(indices)):
        if skip:
            skip = False
            continue

        # 如果发现匹配 pair，就合并成新 token
        if i < len(indices) - 1 and (indices[i], indices[i+1]) == pair:
            output.append(new_index)
            skip = True  # 跳过下一个
        else:
            output.append(indices[i])
    return output



input_text = 'my name is Mia'
vocab, merges = BPETokenization(input=input_text, max_merge_time=15)
print(vocab)
print(merges)