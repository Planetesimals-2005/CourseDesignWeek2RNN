import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# ================= 1. 准备更复杂的文本数据集（情感二分类） =================
docs = [
    # 正面评论 (Label: 1)
    "this movie is great",
    "i love this film",
    "it was an amazing experience",
    "highly recommended for everyone",
    "absolutely beautiful and wonderful",
    "one of the best movies ever",
    "i really liked the acting",
    # 负面评论 (Label: 0)
    "this movie is terrible",
    "i hate this film",
    "it was a waste of time",
    "very boring and too long",
    "the acting was extremely bad",
    "worst experience ever",
    "i do not recommend this movie"
]
labels = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

# ================= 2. 文本分词与序列填充 =================
# 初始化分词器并构建词表
tokenizer = Tokenizer()
tokenizer.fit_on_texts(docs)
sequences = tokenizer.texts_to_sequences(docs)

# 词汇表大小（加 1 是因为索引 0 预留给 padding）
vocab_size = len(tokenizer.word_index) + 1
print(f"【构建成功】词汇表大小为: {vocab_size}")

# 固定句长为 5 词，超出的截断，不足的在后方填充 (padding='post')
max_length = 5
padded_docs = pad_sequences(sequences, maxlen=max_length, padding='post')
print("编码并对齐后的特征矩阵形状:", padded_docs.shape) # 形状为 (14, 5)

# ================= 3. 构建升级版的 RNN 模型 =================
model = Sequential()

# 1. 词嵌入层：把单词索引转化为 8 维的稠密向量
model.add(Embedding(input_dim=vocab_size, output_dim=8, input_length=max_length))

# 2. 循环层：使用 16 个隐藏神经元的 SimpleRNN，具备更强的特征捕捉能力
model.add(SimpleRNN(16, activation='tanh', return_sequences=False))

# 3. 输出层：二分类任务，使用 Sigmoid 激活函数，输出 0~1 之间的概率值
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ================= 4. 增加 Epoch 轮数进行训练 =================
print("\n开始训练...")
history = model.fit(padded_docs, labels, epochs=50, verbose=1)

# ================= 5. 测试模型在未见样本上的泛化能力 =================
print("\n--- 模型测试 ---")
test_docs = [
    "i love this movie",       # 预期输出接近 1 (正面)
    "this film is terrible"    # 预期输出接近 0 (负面)
]
# 用相同的 tokenizer 进行编码和填充
test_sequences = tokenizer.texts_to_sequences(test_docs)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post')

predictions = model.predict(test_padded)

for doc, pred in zip(test_docs, predictions):
    sentiment = '正面' if pred[0] > 0.5 else '负面'
    print(f"文本: '{doc}' -> 预测正面概率: {pred[0]:.4f} (分类结果: {sentiment})")