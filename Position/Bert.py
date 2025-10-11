#!/usr/bin/python
# -*- coding:utf-8 -*-
# Bert Position （可学习的位置编码）
# InputEmbedding = TokenEmbedding + SegmentEmbedding + PositionEmbedding
# TokenEmbedding: 每个词或者wordpiece的向量表示
# SegmentEmbedding：区分每个句子
# PositionEmbedding：表示每个token的位置（0，1，2，……）
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length=512, segment_vocab_size=2):
        super(BertEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, embedding_dim)
        self.position_embed = nn.Embedding(max_length, embedding_dim)
        self.segment_embed = nn.Embedding(segment_vocab_size, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, token_ids, segment_ids):
        # token_ids: [batch_size, seq_len]
        seq_len = token_ids.size(-1)
        # 生成和序列长度一致的位置张量
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand_as(token_ids)

        position_embeddings = self.position_embed(position_ids)
        token_embeddings = self.token_embed(token_ids)
        segment_embeddings = self.segment_embed(segment_ids)

        embeddings = token_embeddings + position_embeddings + segment_embeddings

        return self.layer_norm(embeddings)

class MultiBertAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiBertAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 三个线性变换：Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.size()
        H = self.num_heads

        # Q,K,V: [batch_size, seq_len, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 拆分成多头：[batch_size, H, seq_len, d_model/H]
        Q = Q.view(batch_size, seq_len, H, -1).transpose(1, 2)
        K = K.view(batch_size, seq_len, H, -1).transpose(1, 2)
        V = V.view(batch_size, seq_len, H, -1).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # 转回原形状: [batch_size, seq_len, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        return self.W_o(output)

# 模拟输入：batch_size=2，序列长度=5
batch_size, seq_len = 2, 5
vocab_size = 1000
embed_size = 32
num_heads = 4

# 构造假输入
token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
segment_ids = torch.zeros_like(token_ids)  # 全部属于句子A

# 初始化模块
embedding_layer = BertEmbedding(vocab_size, embed_size, max_length=seq_len)
attention_layer = MultiBertAttention(embed_size, num_heads)

# 前向传播
embeddings = embedding_layer(token_ids, segment_ids)
attention_output = attention_layer(embeddings)

print("🔹 Token IDs:\n", token_ids)
print("\n🔹 Embedding 输出形状:", embeddings.shape)
print("🔹 Attention 输出形状:", attention_output.shape)



