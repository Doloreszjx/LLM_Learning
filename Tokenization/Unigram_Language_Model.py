# 它假设一个句子中的每个单词是独立出现的。也就是说：

# 句子的概率 = 所有单词的概率的乘积。

from collections import Counter

# 一个简单语料库
corpus = ["I love AI", "AI is powerful", "I love machine learning"]

# 统计词频
word_counts = Counter()
for sentence in corpus:
    for word in sentence.lower().split():
        word_counts[word] += 1

# 转换为概率分布
total = sum(word_counts.values())
unigram_probs = {word: count / total for word, count in word_counts.items()}

print(unigram_probs)