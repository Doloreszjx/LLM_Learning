import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

# 示例语料
corpus = [
    "the cat sits on the mat",
    "the dog barks at the cat",
    "the cat drinks milk"
]

# 词表构建
words = " ".join(corpus).split()
vocab = set(words)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)

print("词表：", word2idx)

def generate_cbow_data(corpus, window_size=2):
    data = []
    for sentence in corpus:
        tokens = sentence.split()
        for i in range(window_size, len(tokens) - window_size):
            context = [tokens[i - j - 1] for j in range(window_size)] + \
                      [tokens[i + j + 1] for j in range(window_size)]
            target = tokens[i]
            data.append((context, target))
    return data

data = generate_cbow_data(corpus)
print("样本示例：", data[:3])


class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_idxs):
        # context_idxs: [batch_size, context_len]
        embeds = self.embeddings(context_idxs)  # [batch, context_len, embed_dim]
        hidden = embeds.mean(dim=1)  # 取平均 (bag of words)
        output = self.linear(hidden)  # 预测中心词
        return output


embedding_dim = 10
model = CBOW(vocab_size, embedding_dim)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# 数据准备
def prepare_batch(context_words, target_word):
    context_idxs = torch.tensor([word2idx[w] for w in context_words], dtype=torch.long)
    target_idx = torch.tensor([word2idx[target_word]], dtype=torch.long)
    return context_idxs, target_idx


# 训练循环
for epoch in range(200):
    total_loss = 0
    for context, target in data:
        context_idxs, target_idx = prepare_batch(context, target)
        context_idxs = context_idxs.unsqueeze(0)  # batch 维度

        optimizer.zero_grad()
        output = model(context_idxs)
        loss = loss_fn(output, target_idx)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1} | Loss: {total_loss:.4f}")

# 查看某些词的 embedding
for word in ["cat", "dog", "milk"]:
    # .detach()：脱离计算图，避免影响模型训练；
    emb = model.embeddings.weight[word2idx[word]].detach().numpy()
    print(f"{word}: {emb}")
