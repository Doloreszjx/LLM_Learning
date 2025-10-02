#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import defaultdict

def BPE_testBPE(input, merge_time=10):
    # 1. 先把输入转-》对应编码-》整数（方便匹配
    indices = list(map(int, input.encode('utf-8')))
    print('将输入的string转化成每个字符对应的编码整数', indices)
    # 2. 记录合并规则{(int, int): int} key--token对， value：序列号
    mergeRule = {}
    # 3. 初始词表：单字节
    vocab = {x: bytes([x]) for x in range(256)}

    # 4. 循环执行 BPE 合并
    for i in range(merge_time):
        # 4.1 统计所有相邻 pair 的出现次数
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):
            counts[(index1, index2)] += 1
        # 没有可合并的 pair
        if not counts:
            break
        # 4.2 找出出现次数最多的pair
        pair = max(counts, key=counts.get)
        index1, index2 = pair

        # 4.3 将新的token pair添加到合并规则中
        newIndex = 256 + i
        mergeRule[pair] = newIndex

        # 4.4 更新词表：新token的字节串 = 两个旧token的拼接
        vocab[newIndex] = vocab[index1] + vocab[index2]

        # 4.5 在文本中合并，将（index1， index2） -》 newIndex
        indices = merge(indices, pair, newIndex)

        print("当前序列:", [vocab[idx] for idx in indices])
    print(f'最新词表： {vocab}，\n最新合并规则： {mergeRule}')


def merge(indices, pair, newIndex):
    output = []
    skip = False

    for i in range(len(indices)):
        if skip:
            skip = False
            continue

        if i < len(indices) - 1 and (indices[i], indices[i + 1]) == pair:
            output.append(newIndex)
            skip = True  # 跳过下一个
        else:
            output.append(indices[i])

    return output


input = 'Hi there, how are you?'
BPE_testBPE(input=input, merge_time=20)