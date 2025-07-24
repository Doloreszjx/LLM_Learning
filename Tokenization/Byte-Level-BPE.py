from tokenizers import ByteLevelBPETokenizer
import os

os.makedirs("byte_bpe_demo", exist_ok=True)

# 创建一个临时文本文件作为语料
with open("tiny_corpus.txt", "w", encoding="utf-8") as f:
    f.write("Hello, world!\nI love pizza 🍕\nGPT is amazing 🚀")

# 初始化 tokenizer
tokenizer = ByteLevelBPETokenizer()

# 训练一个小模型（真实训练一般用几百万条数据）
tokenizer.train(
    files="tiny_corpus.txt",
    vocab_size=200,
    min_frequency=1,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
)

# 保存模型（可选）
tokenizer.save_model("byte_bpe_demo")

# 加载刚刚训练的 tokenizer
tokenizer = ByteLevelBPETokenizer(
    "byte_bpe_demo/vocab.json",
    "byte_bpe_demo/merges.txt"
)

# 要测试的句子
text = "Hello, pizza! 🍕🚀"

# 编码成 token
encoded = tokenizer.encode(text)
print("🔢 Token IDs:", encoded.ids)
print("🧩 Tokens:", encoded.tokens)

# 解码回原文
decoded = tokenizer.decode(encoded.ids)
print("🔁 Decoded text:", decoded)
