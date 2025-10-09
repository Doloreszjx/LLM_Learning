from collections import Counter
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 示例语料
corpus = [
    "the cat sits on the mat",
    "the dog barks at the cat",
    "the cat drinks milk"
]

words = ' '.join(corpus).split()
vocab = set(words)
print(vocab)

vocab_size = len(vocab)
word2Idx = {w: i for i, w in enumerate(vocab)}
print('word2Idx', word2Idx)
idx2Word = {i: w for w, i in word2Idx.items()}
print(idx2Word)

# 词频统计
# 每个token出现的次数
word_count = Counter(words)
print(word_count)
# word_freqs = np.array([word_count[idx2Word[i]] for i in range(vocab_size)])
word_freqs = np.array([word_count[w] for w in vocab])
word_freqs = word_freqs / word_freqs.sum()
# Word2Vec 提出的负采样分布 unigram^0.75
word_freqs = word_freqs ** 0.75
word_freqs = word_freqs / word_freqs.sum()
print('负采样后每个token在整体中的出现频率', word_freqs, word_freqs.sum())

# ------- CBOW 数据生成 ---------
def generate_cbow_embedding(corpus, window_size=2):
    data = []
    for sentence in corpus:
        tokens = sentence.split()
        for i in range(window_size, len(tokens)-window_size):
            context = [tokens[i-j-1] for j in range(window_size)] + [tokens[i+j+1] for j in range(window_size)]
            target = tokens[i]
            data.append((context, target))
    return data

data = generate_cbow_embedding(corpus, window_size=2)
# print(data[:3])

# ------ CBOW with Negative Sampling -------
class CBOW_NegSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW_NegSampling, self).__init__()
        self.input_emb = nn.Embedding(vocab_size, embedding_dim)
        self.output_emb = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, context_idxs):
        # context_idxs: [batch_size, context_size]
        embedded = self.input_emb(context_idxs)
        h = embedded.mean(dim=1)
        return h

    def loss(self, h, pos_idxs, neg_idxs):
        # h: [batch_size, embedding_dim]
        # pos_idx: [batch_size]
        # neg_idx: [batch_size, num_neg]
        batch_size = pos_idxs.size(0)
        # 正样本
        # pos_embedded: [batch_size, embedding_dim]
        pos_embedded = self.output_emb(pos_idxs)
        pos_score = torch.sum(pos_embedded * h, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        # 负样本
        # neg_embedded: [batch_size, num_neg, embedding_dim]
        neg_embedded = self.output_emb(neg_idxs)
        # h_unsq: [batch_size, 1, embedding_dim]
        h_unsq = h.unsqueeze(1)
        # neg_score: [batch_size, num_neg]
        neg_score = torch.bmm(neg_embedded, h_unsq.transpose(1, 2)).squeeze()
        neg_loss = F.logsigmoid(-neg_score).sum(1)

        return -(pos_loss + neg_loss).mean()

# -------- 训练参数 --------
embedding_dim = 10
num_neg = 5
batch_size = 4
epochs = 150
lr = 0.1

def get_negative_samples(batch_size, num_neg):
    # 用 np.random.choice 按词频概率（word_freqs）从词汇表（vocab_size）选负例，
    # 生成 [batch_size, num_neg] 的索引矩阵
    neg_samples = np.random.choice(vocab_size, size=[batch_size, num_neg], p=word_freqs)

    return torch.tensor(neg_samples, dtype=torch.long)

def prepare_batch(data, start, batch_size):
    batch_data = data[start:start+batch_size]
    context_idxs = []
    target_idxs = []
    for context, target in batch_data:
        context_idxs.append([word2Idx[w] for w in context])
        target_idxs.append(word2Idx[target])
    return torch.tensor(context_idxs, dtype=torch.long), torch.tensor(target_idxs, dtype=torch.long)

model = CBOW_NegSampling(vocab_size, embedding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# Training Loop
# -------------------
for epoch in range(epochs):
    total_loss = 0
    random.shuffle(data)
    for i in range(0, len(data), batch_size):
        context_idxs, pos_idx = prepare_batch(data, i, batch_size)
        neg_idx = get_negative_samples(context_idxs.size(0), num_neg)

        optimizer.zero_grad()
        h = model(context_idxs)
        loss = model.loss(h, pos_idx, neg_idx)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * context_idxs.size(0)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(data):.4f}")









