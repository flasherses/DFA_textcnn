# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:37:26 2024

@author: Administrator
"""
import os
import pickle
import pandas as pd
import joblib
import jieba as jieba
from tensorflow.keras.preprocessing.text import Tokenizer

import tensorflow as tf
# from gensim.models.word2vec import Word2Vec

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # 设置GPU分配器为cuda_malloc_async
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # 允许GPU内存增长

gpus = tf.config.experimental.list_physical_devices('GPU')  # 列出所有可用的GPU设备
tf.config.experimental.set_memory_growth(gpus[0],True)  # 设置第一个GPU设备的内存增长

# 读取命中数据和非命中数据
hit_data = pd.read_csv(r'data/cleaned_censor_hit.csv')
non_hit_data = pd.read_csv(r'data/cleaned_comment_tb.csv')

hit_data['label'] = 1  # 命中数据标签为1
non_hit_data['label'] = 0  # 非命中数据标签为0

# 合并命中数据和非命中数据
data = pd.concat([hit_data, non_hit_data])

# 读取人工删除的数据并合并
mg1=pd.read_csv(r'data/人工删除.csv')
mg1.columns=['id','content','label']
mg1=mg1[['content','label']]
data = pd.concat([data, mg1])

# 读取敏感词1并合并
with open(r'data/敏感词1.txt', 'r',encoding='utf8') as file:
    content_string = file.read()
mg2 = pd.DataFrame([ line for line in content_string.split('|')],columns=['content'])
mg2['label'] = 1
data = pd.concat([data, mg2])

# 读取敏感词2并合并
with open(r'data/敏感词2.txt', 'r',encoding='gbk') as file:
    content_string = file.read()
mg3 = pd.DataFrame([ line for line in content_string.split('|')],columns=['content'])
mg3['label'] = 1
data = pd.concat([data, mg3])

# 读取敏感词3并合并
with open(r'data/敏感词3.txt', 'r',encoding='gbk') as file:
    content_string = file.read()
mg4 = pd.DataFrame([ line for line in content_string.split('\n')],columns=['content'])
mg4['label'] = 1
data = pd.concat([data, mg4])

# 删除重复的内容
data.drop_duplicates(subset='content', inplace=True)

# 打印敏感词的总数和数据的总数
print(mg2.shape[0]+mg3.shape[0]+mg4.shape[0]+1793)
print(data.shape[0])
# merged_df = pd.concat([mg2, mg3, mg4])
# filtered_df = merged_df[merged_df['content'].apply(lambda x: len(str(x)) >= 2)]
# filtered_df['content'].to_csv('custom_dict.txt', header=False, index=False, mode='w', sep='\n')

def chinese_word_cut(text):
    if len(text)<4:
        return text
    text = " ".join(jieba.cut(text))
    
    return text
    # return ''.join(lazy_pinyin(text))

def remove_stopwords(text):
    stopwords = set()
    with open(r'data/chinese_stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip())
    words = [word for word in text.split() if word not in stopwords]
    return words

data['content'] = data['content'].apply(chinese_word_cut)
data['content'] = data['content'].apply(remove_stopwords)

mask = data['content'].apply(lambda x: len(x) <= 60)
long = data['content'].apply(lambda x: len(x) >= 60)

print(data)

data.to_csv('data/processed_data.csv', encoding='utf-8-sig', index=False)

tokenizer = Tokenizer()  # 创建一个Tokenizer对象
tokenizer.fit_on_texts(data['content'])  # 使用数据中的文本来训练Tokenizer
vocab = tokenizer.word_index  # 获取词汇表

with open('model/vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

import joblib
joblib.dump(tokenizer, 'data/tokenizer.pkl')  # 将训练好的Tokenizer对象保存到文件中

data = data[mask]