from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras import optimizers
import numpy as np
from keras import regularizers
from keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
tf.keras.losses.sparse_categorical_crossentropy()


# 拆分训练集和测试集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# plt.imshow(np.concatenate((train_images[0], train_images[1]), axis=1))
# plt.show()

# 从训练集中提取出验证集
val_images, val_labels = train_images[:10000], train_labels[:10000]
train_images, train_labels = train_images[10000:], train_labels[10000:]

# 转成向量
train_images = train_images.reshape((50000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))
val_images = val_images.reshape((10000, 28 * 28))

# 将labels转为one-hot向量
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
val_labels = to_categorical(val_labels)

# 归一化
# train_images = train_images.astype('float32') / 255

# 搭建网络
network = models.Sequential()
network.add(layers.Dense(512, activation='sigmoid', input_shape=(28 * 28,))) # relu
network.add(layers.Dense(10, activation='softmax'))

# 编译（主要用于配置训练算法/优化算法）
network.compile(optimizer=optimizers.sgd(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练
history = network.fit(train_images, train_labels, epochs=5, batch_size=128, validation_data=(val_images, val_labels))

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

# 测试
test_loss, test_acc = network.evaluate(test_images, test_labels)

# 作图
epochs = range(1, len(loss_values) + 1)
fig, axes = plt.subplots(2,1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')


network.predict()