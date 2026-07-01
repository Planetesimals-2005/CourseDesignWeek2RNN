import os
import random
import numpy as np
import tensorflow as tf
import matplotlib

# 使用 Agg 后端，避免 PyCharm/无界面环境弹窗报错
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, GRU, Dense, Dropout, SpatialDropout1D
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint



# =====================================================
# 0. 固定随机种子：尽量保证每次运行结果可复现
# =====================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 可选：进一步提高可复现性
os.environ["PYTHONHASHSEED"] = str(SEED)



# =====================================================
# 1. 参数配置
# =====================================================
MAX_FEATURES = 20000     # 只保留最常见的 20000 个词
MAX_LEN = 200            # 每条评论统一截断/填充到 200 个词
EMBED_DIM = 64           # 词向量维度
BATCH_SIZE = 128
EPOCHS = 10              # 设置大一点，交给 EarlyStopping 自动停止


# =====================================================
# 2. 加载并预处理 IMDB 数据集
# =====================================================


print("正在加载 IMDB 文本数据集...")

from keras.layers import Input

model = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(input_dim=MAX_FEATURES, output_dim=EMBED_DIM),
    SpatialDropout1D(0.25),
    GRU(64, dropout=0.3, recurrent_dropout=0.2),
    Dropout(0.4),
    Dense(1, activation="sigmoid")
])


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)

# 将不同长度的评论统一处理成固定长度
x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")

print("训练集形状:", x_train.shape)
print("测试集形状:", x_test.shape)


# =====================================================
# 3. 构建模型
#    GRU 比 SimpleRNN 更适合处理较长文本序列
# =====================================================
model = Sequential([
    # 将整数词 ID 映射为稠密词向量
    Embedding(
        input_dim=MAX_FEATURES,
        output_dim=EMBED_DIM,
    ),

    # 对词向量做整体 dropout，减少模型依赖某些特定词
    SpatialDropout1D(0.25),

    # GRU 层：比 SimpleRNN 更能捕捉长距离上下文信息
    GRU(
        units=64,
        dropout=0.3,
        recurrent_dropout=0.2
    ),

    # 全连接前再做 dropout，进一步缓解过拟合
    Dropout(0.4),

    # 二分类输出层，sigmoid 输出正面评论概率
    Dense(1, activation="sigmoid")
])


# =====================================================
# 4. 编译模型
# =====================================================
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

model.compile(
    optimizer=optimizer,
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\n--- 模型结构摘要 ---")
model.summary()


# =====================================================
# 5. 回调函数：防止过拟合
# =====================================================

# 当验证集 loss 连续 2 轮不下降时提前停止
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=1,
    restore_best_weights=True,
    verbose=1
)

# 保存验证集表现最好的模型
checkpoint = ModelCheckpoint(
    filepath="best_imdb_gru_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# =====================================================
# 6. 训练模型
# =====================================================
print("\n--- 开始训练 GRU 模型 ---")

history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    callbacks=[early_stop, checkpoint],
    verbose=1
)


# =====================================================
# 7. 在测试集上评估最终模型
# =====================================================
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print("\n" + "=" * 45)
print(f"测试集 Loss: {test_loss:.4f}")
print(f"测试集 Accuracy: {test_acc:.4f}")
print("=" * 45)


# =====================================================
# 8. 绘制训练曲线
# =====================================================

# 根据实际训练轮数自动生成横坐标
epochs_range = range(1, len(history.history["loss"]) + 1)

# ---------- Loss 曲线 ----------
plt.figure(figsize=(8, 4))
plt.plot(epochs_range, history.history["loss"], label="Train Loss", linewidth=2)
plt.plot(epochs_range, history.history["val_loss"], label="Validation Loss", linewidth=2)
plt.title("GRU Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(epochs_range)
plt.legend()
plt.grid(True, linestyle="--")
plt.tight_layout()
plt.savefig("imdb_gru_loss.png", dpi=300)
plt.close()

print("损失曲线图已保存为 imdb_gru_loss.png")


# ---------- Accuracy 曲线 ----------
plt.figure(figsize=(8, 4))
plt.plot(epochs_range, history.history["accuracy"], label="Train Accuracy", linewidth=2)
plt.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy", linewidth=2)
plt.title("GRU Training & Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xticks(epochs_range)
plt.legend()
plt.grid(True, linestyle="--")
plt.tight_layout()
plt.savefig("imdb_gru_accuracy.png", dpi=300)
plt.close()

print("准确率曲线图已保存为 imdb_gru_accuracy.png")
