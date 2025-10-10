#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """
        q: [batch_size, h, seq_len, d_model]
        k: [B, h, T_k, d_k]
        v: [B, h, T_k, d_k]
        mask: None or [B, 1, T_q, T_k] or broadcastable to[B, h, T_q, T_k]
        returns:
        out: [B, h, T_q, d_k]
        attn: [B, h, T_q, T_k]
    """
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    att = F.softmax(scores, dim=-1)

    if dropout is not None:
        att = dropout(att)

    output = torch.matmul(att, v)
    return output, att

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        def split_heads(tensor):
            return tensor.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        Q, K, V = map(split_heads, (Q, K, V))
        print(Q.shape, K.shape, V.shape)

        out, att = scaled_dot_product_attention(Q, K, V)
        # out: [B, h, T_q, d_k]
        # attn: [B, h, T_q, T_k]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        return out, att

# 模拟句子：["The", "cat", "sat", "there"]
vocab = ["The", "cat", "sat", "there"]
embedding_dim = 16
num_heads = 4

# 随机初始化 embedding（假设 embedding 已训练好）
embeddings = nn.Embedding(len(vocab), embedding_dim)
inputs = torch.tensor([[0, 1, 2, 3]])  # batch=1, seq_len=4
x = embeddings(inputs)

print("Input shape:", x.shape)  # [1, 4, 16]

mha = MultiHeadAttention(d_model=embedding_dim, num_heads=num_heads)
output, attn_weights = mha(x)

print("Output shape:", output.shape)        # [1, 4, 16]
print("Attention weights shape:", attn_weights.shape)  # [1, 4, 4, 4] (B, heads, T, T)

# 打印第一个head的注意力权重
head = 0
print(f"\nHead {head} Attention Matrix:")
print(torch.round(attn_weights[0, head], decimals=2))

import matplotlib.pyplot as plt
import seaborn as sns

sns.heatmap(attn_weights[0, 0].detach().numpy(), cmap="Blues", annot=True)
plt.xlabel("Key (be viewed)")
plt.ylabel("Query (current token)")
plt.title("Head 0 Attention Heatmap")
plt.show()






