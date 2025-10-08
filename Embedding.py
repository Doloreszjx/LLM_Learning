#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

vocab = ['apple', 'banana', 'cherry']
# 构造一个词到索引的映射
word2Idx = {word: idx for idx, word in enumerate(vocab)}

print('word2Idx: ', word2Idx)

def ont_hot_encoding(word, vocab, word2Idx):
    # 创建一个长度与词汇表相同的全0向量
    encoding = torch.zeros(len(vocab))
    # 获得该词的索引，默认返回-1
    idx = word2Idx.get(word, -1)
    if idx != -1:
        encoding[idx] = 1
    return encoding

print('------------ ONE HOT ENCODING ------------')

for word in vocab:
    print(f'word: {word}, encoding: {ont_hot_encoding(word, vocab, word2Idx)}')

print('可以看到，onehot编码的问题在于，生成的向量之间没有什么关系，随之词的增加维度会非常的大，且稀疏')