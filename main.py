import numpy as np
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

# ================= 1. 数据定义与编码 =================
docs = ['this, is', 'is an']
labels = ['an', 'example']

counts = Counter()
for i, review in enumerate(docs + labels):
    counts.update(review.split())

# 按照词频排序
words = sorted(counts, key=counts.get, reverse=True)
vocab_size = len(words)
word_to_int = {word: i for i, word in enumerate(words, 1)}

print("【词表映射】:", word_to_int)

# 编码文本
encoded_docs = []
for doc in docs:
    encoded_docs.append([word_to_int[word] for word in doc.split()])

# 编码标签
encoded_labels = []
for label in labels:
    encoded_labels.append([word_to_int[word] for word in label.split()])

print('encoded_docs: ', encoded_docs)
print('encoded_labels: ', encoded_labels)

# 修正点 1：将二维 encoded_labels 扁平化为一维 [2, 4]，
# 以避免 to_categorical 生成不匹配的三维张量形状 (2, 1, 5)
flat_encoded_labels = [label[0] for label in encoded_labels]

max_length = 2
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='pre')

# 将标签转换为 one-hot 编码，目标维度为 (2, 5)
one_hot_encoded_labels = to_categorical(flat_encoded_labels, num_classes=5)
print("【目标 One-hot 标签】:\n", one_hot_encoded_labels)

# 改变输入维度为 (batch_size, timesteps, input_dim) -> (2, 2, 1)
padded_docs = padded_docs.reshape(2, 2, 1)

# ================= 2. 构建与训练 RNN 模型 =================
embed_length = 1
max_length = 2

model = Sequential()
model.add(SimpleRNN(1, activation='tanh',
                    return_sequences=False,
                    input_shape=(max_length, embed_length),
                    unroll=True))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

# 训练模型
model.fit(padded_docs, np.array(one_hot_encoded_labels), epochs=100)

# 预测结果
pred = model.predict(padded_docs[0].reshape(1, 2, 1))
print("\n【模型预测输出】:\n", pred)

# ================= 3. 手动前向传播计算与验证 =================
weights = model.get_weights()
# weights 中的结构:
# weights[0]: 输入到隐藏层的权重 W_xh
# weights[1]: 循环权重 W_hh
# weights[2]: 隐藏层的偏置 b_h
# weights[3]: 隐藏层到输出层的权重 W_hy
# weights[4]: 输出层的偏置 b_y

print("\n--- 开始手动前向传播计算验证 ---")
print("测试输入 padded_docs[0]:\n", padded_docs[0])

# Time Step 0
input_t0 = padded_docs[0][0]
input_t0_kernel_bias = input_t0 * weights[0] + weights[2]
hidden_layer0_value = np.tanh(input_t0_kernel_bias)

# Time Step 1
input_t1 = padded_docs[0][1]
input_t1_kernel_bias = input_t1 * weights[0] + weights[2]
input_t1_recurrent = hidden_layer0_value * weights[1]
total_input_t1 = input_t1_kernel_bias + input_t1_recurrent
output_t1 = np.tanh(total_input_t1)

# 密集层输出
final_output = output_t1 * weights[3] + weights[4]

# 手动 Softmax 计算
manual_softmax = np.exp(final_output) / np.sum(np.exp(final_output))
print("【手动计算的 Softmax 输出】:\n", manual_softmax)