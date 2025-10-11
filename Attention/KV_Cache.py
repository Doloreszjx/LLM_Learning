#!/usr/bin/python
# -*- coding:utf-8 -*-
# 在推理阶段，自注意力机制只能看到之前推理出的词，因此每次推理新的内容，都需要重新计算KV，耗时
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
class Attention_KVCache(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Attention_KVCache, self).__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, cache=None):
        # x: [batch_size, seq_len, d_model]
        B, S, D = x.size()
        H = self.num_heads

        # Q,K,V: [batch_size, seq_len, d_model]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # reshape -> [batch_size, H, seq_len, d_model/H]
        Q = Q.view(B, S, H, -1).transpose(1, 2)
        K = K.view(B, S, H, -1).transpose(1, 2)
        V = V.view(B, S, H, -1).transpose(1, 2)

        if cache is not None:
            # 将K，V信息拼接在序列长度
            K = torch.cat([cache['k'], K], dim=2)
            V = torch.cat([cache['v'], V], dim=2)

        # attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = torch.softmax(scores, dim=-1)
        out = torch.matmul(att, V)
        # [batch_size, H, seq_len, d_model/H]
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.layer_norm(out)

        new_cache = {"k": K, "v": V}
        return out, new_cache

torch.manual_seed(42)
attention = Attention_KVCache(d_model=16, num_heads=4)

# 模拟输入三个时间步（单步输入）
x1 = torch.randn(1, 1, 16)
x2 = torch.randn(1, 1, 16)
x3 = torch.randn(1, 1, 16)

cache = None
for step, x in enumerate([x1, x2, x3], start=1):
    out, cache = attention(x, cache)
    print(f"Step {step}: cache K shape = {cache['k'].shape}, output shape = {out.shape}")


