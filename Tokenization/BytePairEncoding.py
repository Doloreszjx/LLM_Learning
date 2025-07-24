# Byte Pair Encoding (BPE) 实现
# 1. 将词表中的每个词拆成字符，并在结尾添加一个特殊符号 "</w>"。
# 2. 计算词表中相邻字符对的频率。
# 3. 找到频率最高的字符对，将它们合并成一个新符号。


from collections import defaultdict, Counter

# 初始词表（每个词拆成字符 + 结尾符）
corpus = ["low", "lowest", "newest", "wider"]
tokens = [list(word) + ["</w>"] for word in corpus]

# 把词表变成计数字典
def get_vocab(tokens):
  vocab = defaultdict(int)
  for token in tokens:
    vocab[" ".join(token)] += 1
  return vocab

# 计算相邻字符对的频率
def get_freq(vocab):
  pairs = defaultdict(int)
  for word, freq in vocab.items():
    symbols = word.split()
    for i in range(len(symbols) - 1):
      pair = (symbols[i], symbols[i + 1])
      pairs[pair] += freq
  return pairs

# 合并字符对
def merge_vocab(vocab, pair_to_merge):
  print(f"合并字符对: {pair_to_merge}")
  new_vocab = {}
  a, b = pair_to_merge
  target = " ".join([a, b])
  replacement = a + b
  for word in vocab:
    # 将a b替换为ab
    new_word = word.replace(target, replacement)
    new_vocab[new_word] = vocab[word]
  return new_vocab

vocab = get_vocab(tokens)
print("初始词表:", vocab)

for i in range(5):
  pairs = get_freq(vocab)
  if not pairs:
    print("没有更多的字符对可以合并了。")
    break
  most_common_pair = max(pairs, key=pairs.get)
  vocab = merge_vocab(vocab, most_common_pair)
  print(f"第{i+1}次合并后的词表:", vocab)

