# 和BPE非常的像，只是 WordPiece 是基于字符的子词分割方法，但是是基于最大似含函数作为频率

from tokenizers import BertWordPieceTokenizer
import os

os.makedirs("wordpiece/wordpiece_demo", exist_ok=True)

# 1. 初始化 tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=True,
    lowercase=True,
    strip_accents=True
)

# 2. 训练 tokenizer，准备一份小文本语料
with open("sample.txt", "w") as f:
    f.write("unbelievable unbelievable believable unknown")

# 3. 使用语料训练 WordPiece 模型
tokenizer.train(
    files=["sample.txt"],
    vocab_size=30,             # 词表大小
    min_frequency=1,           # 一个词至少出现一次才会被考虑
    limit_alphabet=1000,       # 限制初始字母表大小（如字符数）
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# 4. 保存模型（可选）
tokenizer.save_model("wordpiece/wordpiece_demo")

# 5. 使用 tokenizer 编码
encoded = tokenizer.encode("unbelievable unknownly")
print("Tokens:", encoded.tokens)
print("IDs:", encoded.ids)

Tokens: ['un', '##believable', 'unknown', '##ly']
IDs:   [10, 14, 20, 22]
# 6. 解码回原文
decoded = tokenizer.decode(encoded.ids) # 将编码后的 ID（即 token 的编号）转换回原始的文本字符串
print("Decoded text:", decoded)

