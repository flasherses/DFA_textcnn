# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:37:26 2024

@author: Administrator
"""
import os

import pickle
import pandas as pd
import numpy as np

import joblib
import jieba as jieba


from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from models import t2v,TextCNN_model_1
# from gensim.models.word2vec import Word2Vec

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # 设置GPU分配器为cuda_malloc_async
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # 允许GPU内存增长

gpus = tf.config.experimental.list_physical_devices('GPU')  # 列出所有可用的GPU设备
tf.config.experimental.set_memory_growth(gpus[0],True)  # 设置第一个GPU设备的内存增长




# 加载Tokenizer对象
tokenizer = joblib.load('model/tokenizer.pkl')
vocab=tokenizer.word_index

embedding_matrix = t2v(vocab)

# 读取处理后的数据
data = pd.read_csv('data/processed_data.csv')
mask = data['content'].apply(lambda x: len(x) <= 60)
data = data[mask]

X_train, X_test, y_train, y_test = train_test_split(data['content'], data['label'], test_size=0.2, random_state=42)
x_train_word_ids = tokenizer.texts_to_sequences(X_train)
x_test_word_ids = tokenizer.texts_to_sequences(X_test)

x_train_padded_seqs = pad_sequences(x_train_word_ids, maxlen=60, padding='post', value=1e-10) #将超过固定值的部分截掉，不足的在最前面用0填充
x_test_padded_seqs = pad_sequences(x_test_word_ids, maxlen=60, padding='post', value=1e-10)



model=TextCNN_model_1(vocab,embedding_matrix)
# 打印模型结构信息
model.summary()
    # 训练模型

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    verbose=1,
    patience=3,
    mode='max',
    restore_best_weights=True)

history = model.fit(
    x_train_padded_seqs, y_train, batch_size=128, epochs=100, 
    validation_split=0.3,
    callbacks = [early_stopping],)


result = model.predict(x_test_padded_seqs)  # 预测样本属于每个类别的概率
result_labels = np.argmax(result, axis=1)  # 获得最大概率对应的标签
y_predict = result_labels
print('准确率', metrics.accuracy_score(y_test, y_predict))
print('平均f1-score:', metrics.f1_score(y_test, y_predict, average='weighted'))
print('报告', metrics.classification_report(y_test, y_predict))
model.save("model/fasttext_textcnn.h5")
print("***********************************************")

# 全部数据进行训练
# data = data.sample(frac=1).reset_index(drop=True)  # 随机打乱数据并重置索引
# word_ids = tokenizer.texts_to_sequences(data['content'])  # 将文本转换为词汇表中的索引序列
# padded_seqs = pad_sequences(word_ids, maxlen=60, padding='post', value=1e-10)  # 对序列进行填充，使其长度统一为60


# checkpoint_path = "model/best_model.h5"
# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_path,
#     monitor='val_loss',  # 监控验证集的准确率
#     verbose=1,  # 打印保存模型的详细信息
#     save_best_only=True,  # 只保存最佳模型
#     mode='min'  # 最大化监控指标
# )

# # 定义EarlyStopping回调
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     verbose=1,
#     patience=3,
#     mode='min',
#     restore_best_weights=True  # 当早停时，恢复最佳权重
# )

# # 训练模型，并使用回调
# history = model.fit(
#     padded_seqs, data['label'],
#     batch_size=128,
#     epochs=100,
#     validation_split=0.3,
#     callbacks=[early_stopping, model_checkpoint_callback]  # 使用回调列表
# )
# #

# print("***********************************************")

# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_accuracy', 
#     verbose=1,
#     patience=3,
#     mode='max',
#     restore_best_weights=True)

# history = model.fit(
#     padded_seqs, data['label'], batch_size=128, epochs=100, 
#     validation_split=0.3,
#     callbacks = [early_stopping],)