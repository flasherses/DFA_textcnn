import numpy as np
import jieba as jieba

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding,concatenate
from tensorflow.keras.layers import Conv1D,GlobalMaxPooling1D, Dense, Dropout

import fasttext




def t2v(vocab):
    # 加载FastText预训练模型
    fasttext_model = fasttext.load_model(r"pretrain/cc.zh.300.bin")

    # 初始化嵌入矩阵，所有元素初始化为1e-10
    embedding_matrix = np.full((len(vocab) + 1, 300), 1e-10)
    
    # 遍历词汇表中的每个词
    for word, i in vocab.items():
        try:
            # 如果词在预训练的词向量中出现，则用实际的词向量替换
            embedding_vector = fasttext_model[str(word)]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            # 如果词不在预训练的词向量中，使用相近词向量代替
            sim_word = fasttext_model.get_nearest_neighbors(str(word))[0][1]
            embedding_vector = fasttext_model.get_word_vector(sim_word)
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def TextCNN_model_1(vocab,embedding_matrix):
    maxlen=60
    num_filters = 256  # 卷积核数量
    filter_sizes = [3, 4, 5]  # 卷积核尺寸
    dropout_rate = 0.5# Dropout 比例
    input = Input(shape=(maxlen,))
    model = Embedding(len(vocab) + 1, 300, input_length=60, weights=[embedding_matrix], trainable=False)(input)  # 嵌入层
    model = Dropout(dropout_rate)(model)  # Dropout 层
    
    # 多个卷积层和最大池化层
    conv_layers = []
    for filter_size in filter_sizes:
        conv_layer = Conv1D(num_filters, filter_size, padding='same', strides=1, activation='relu')(model)
        pool_layer = GlobalMaxPooling1D()(conv_layer)
        #normalized_layer = BatchNormalization()(pool_layer)
        conv_layers.append(pool_layer)
    
    # 将池化后的特征进行拼接
    merged = concatenate(conv_layers, axis=1)
    model = Dense(128, activation='relu')(merged)  # 全连接层
    #model = BatchNormalization()(model)
    model = Dropout(dropout_rate)(model)  # Dropout 层
    
    output = Dense(2, activation='softmax')(model)  # 输出层
    model = Model(input, output)
    # 指定优化器和损失函数
    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])
    return model
