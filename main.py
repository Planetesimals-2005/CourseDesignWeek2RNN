import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ==========================================
# 1. 文本数据预处理
# ==========================================
docs = ['this, is', 'is an']
labels = ['an', 'example']

# 构建词频字典
counts = Counter()
for i, review in enumerate(docs + labels):
    counts.update(review.split())

# 按照词频排序构建词汇表（从 1 开始编码）
words = sorted(counts, key=counts.get, reverse=True)
word_to_int = {word: i for i, word in enumerate(words, 1)}
print("【词汇映射表】:", word_to_int)

# 将文本转换为整数序列
encoded_docs = [[word_to_int[word] for word in doc.split()] for doc in docs]
encoded_labels = [[word_to_int[word] for word in label.split()] for label in labels]
print('【编码后的文档】: ', encoded_docs)
print('【编码后的标签】: ', encoded_labels)

# 填充序列（设置最大长度为 2）
max_length = 2
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='pre')

# 标签独热编码（One-Hot Encoding），总共 5 个类别（0 到 4）
one_hot_encoded_labels = to_categorical(encoded_labels, num_classes=5)

# 将输入特征重塑为 RNN 需要的格式: (样本数, 时间步, 特征数)
padded_docs = padded_docs.reshape(2, max_length, 1)

# ==========================================
# 2. 构建与训练 RNN 模型
# ==========================================
embed_length = 1

model = Sequential([
    # SimpleRNN 层：1个神经元，tanh 激活函数，unroll=True 展开计算
    SimpleRNN(1, activation='tanh', return_sequences=False, input_shape=(max_length, embed_length), unroll=True),
    # 全连接层：5个神经元，对应 5 个分类的概率
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print("\n--- 模型结构摘要 ---")
model.summary()

print("\n--- 开始模型训练 ---")
model.fit(padded_docs, np.array(one_hot_encoded_labels), epochs=100, verbose=0) # 隐藏繁琐的训练日志
print("训练完成！")

# ==========================================
# 3. 模型预测与底层数学原理手动验证
# ==========================================
print("\n--- 预测与数学验证 ---")
# Keras 模型的原生预测结果
keras_prediction = model.predict(padded_docs[0].reshape(1, max_length, 1), verbose=0)
print("【Keras 模型预测结果】:\n", keras_prediction)

# 获取模型训练后的权重参数
weights = model.get_weights()
W_x = weights[0]  # 输入到隐藏层的权重
W_h = weights[1]  # 循环隐藏层之间的权重
b_h = weights[2]  # 隐藏层偏置
W_d = weights[3]  # 全连接层的权重
b_d = weights[4]  # 全连接层偏置

# 提取第一个文档的数据
doc_0 = padded_docs[0]

# 时间步 t=0 的计算
input_t0 = doc_0[0]
input_t0_kernel_bias = input_t0 * W_x + b_h
hidden_layer0_value = np.tanh(input_t0_kernel_bias)

# 时间步 t=1 的计算
input_t1 = doc_0[1]
input_t1_kernel_bias = input_t1 * W_x + b_h
input_t1_recurrent = hidden_layer0_value * W_h
total_input_t1 = input_t1_kernel_bias + input_t1_recurrent
output_t1 = np.tanh(total_input_t1)

# 全连接层与 Softmax 映射
final_output = output_t1 * W_d + b_d
manual_prediction = np.exp(final_output) / np.sum(np.exp(final_output))

print("【手动数学推演结果】:\n", manual_prediction)