import random
import numpy as np
import tensorflow as tf

# # ==========================================
# # 锁死全局随机种子（确保每次重跑结果完全一致）
# # ==========================================
# random.seed(42)
# np.random.seed(42)
# tf.random.set_seed(42)

# 配置 Matplotlib 只负责静默画图，不触发 PyCharm 插件弹窗报错
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences

# ==========================================
# 1. 解锁更大的数据集视野
# ==========================================
max_features = 20000  # 词汇表从 1万 扩大到 2万
maxlen = 200          # 句子截断长度从 100 扩大到 200 个词

print("正在加载 IMDB 文本数据集...")
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# 填充和对齐序列
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

# ==========================================
# 2. 构建抗过拟合的双层堆叠 RNN 网络
# ==========================================
model = Sequential([
    # 词嵌入向量维度从 32 提升到 64，容纳更复杂的语义空间
    Embedding(input_dim=max_features, output_dim=64),

    # 第一层 RNN：设置 return_sequences=True 以便将时序特征完好地传给下一层
    # 同时引入 dropout，强力斩断死记硬背的过拟合倾向
    SimpleRNN(64, activation='tanh', return_sequences=True, dropout=0.2),

    # 第二层 RNN：聚合时序特征，收拢为单个向量
    SimpleRNN(32, activation='tanh', return_sequences=False, dropout=0.2),

    # 输出层
    Dense(1, activation='sigmoid')
])

# 稍微调低学习率，让模型在吞噬大数据量时收敛得更稳健
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print("\n--- 深度 RNN 模型结构摘要 ---")
model.summary()

# ==========================================
# 3. 开始训练模型
# ==========================================
print("\n--- 开始训练 RNN ---")
# 适当增大 batch_size（从 64 到 128），配合更大的文本信息量
history = model.fit(x_train, y_train, epochs=4, batch_size=128, validation_split=0.2)

# 4. 在完全未知的测试集上评估最终效果
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("\n" + "="*40)
print(f"【测试集最终真实准确率】: {test_acc:.4f}")
print("="*40)

# ==========================================
# 5. 自动绘制并保存无瑕疵的 Loss 曲线图
# ==========================================
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss', color='#1f77b4', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='#d62728', linewidth=2)
plt.title('Pro RNN Training & Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--')
plt.savefig('imdb_rnn_loss.png', dpi=300)
print("\n损失曲线图已保存为 'imdb_rnn_loss.png'")