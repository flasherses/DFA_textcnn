# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:35:39 2024

@author: Administrator
"""
import json
import re
import time

import joblib
import numpy as np
import pymysql
import requests
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import chinese_word_cut, remove_stopwords, replace_word, chinese_word_cut_long, remove_stopwords_long, \
    smooth_split
from DFA import DFAFilter

gfw = DFAFilter()
path="data/processed_data_content.csv"
gfw.parse(path)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# 发送markdown消息
def send_md(webhook, content):
    header = {
        "Content-Type": "application/json",
        "Charset": "UTF-8"
    }
    data = {

        "msgtype": "markdown",
        "markdown": {
            "content": content,
            "mentioned_mobile_list": ["13800001111"]
        }
    }
    data = json.dumps(data)
    info = requests.post(url=webhook, data=data, headers=header)


# MySQL数据库连接配置
db_config = {
    'host': '10.14.1.208',
    'user': 'ops_read',
    'password': 'GW09iOc#iO8*s3mv',
    'database': 'forumwd_db',
    'charset': 'utf8mb4',
    'autocommit': True
    # 'cursorclass': pymysql.cursors.DictCursor
}

# 创建数据库连接
connection = pymysql.connect(**db_config)

webhook = 'https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=970e797a-bb0e-4c1b-ad10-681e32751814'

# 加载模型
tokenizer = joblib.load('model/tokenizer.pkl')
vocab = tokenizer.word_index
model = load_model("model/fasttext_textcnn.h5")

# try:
# 创建一个游标对象
file = open('data/id_file.txt', 'r+')

id = int(file.read())  # 读取文件中的id
previous_minute = time.localtime(time.time()).tm_min  # 获取当前时间的分钟数

with connection.cursor() as cursor:
    while 1:
        # # 执行SQL查询，查找指定id的评论内容
        # sql_query = f"select content from forumwd_censor_hit_content_tb where  content_id={id} and content_type='comment'"
        # cursor.execute(sql_query)
        # r = cursor.fetchall()
        # if (len(r)) == 0:
        #     # 如果上述查询没有结果，则查询另一个表中的内容
        #     sql_query = f"select content from forumwd_comment_tb where  id={id}"
        #     cursor.execute(sql_query)
        #     r = cursor.fetchall()

        #     if (len(r)) == 0:
        #         # 如果仍然没有结果，则打印当前时间和当前扫描的id，并等待3秒后继续
        #         current_minute = time.localtime(time.time()).tm_min
        #         if current_minute != previous_minute:
        #             current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        #             print(f"暂无新数据，时间：{current_time}，当前扫描id：{id}")
        #             previous_minute = current_minute
        #         time.sleep(3)
        #         continue
        # # 如果查询到结果，则更新id，重写文件中的id，并处理查询到的内容
        # # 更新id并重写文件中的id
        # id += 1
        # file.seek(0)
        # file.truncate()
        # file.write(str(id))
        # file.flush()
        # # 获取查询到的结果
        # result = r[0][0]

        result = "<p>那可能是我主动点的别人交易。因为我想交易给小号一点钱就试了试。。但是没注意是刷道队伍交易的小号还是小号交易的队伍中的交易成功了</p>"
        # 移除结果中的HTML标签
        md = (str(result).replace('<p>', '').replace('</p>', ''))
        # 初始化输入文本
        input_text = result
        # 移除输入文本中的数字和空格
        input_text = re.sub(r'\d+', '', input_text)
        input_text = input_text.replace(' ', '')

        DFA_result = gfw.filter(input_text)

        # 初始化标志位
        flag = 0
        if DFA_result:
            print("DFA_result:", DFA_result)
            print(md)
            send_md(webhook,
                        content=f'**帖子内容：**<font color="warning">{md}</font>\n**敏感度：**<font color="warning">1（敏感）</font>\n**敏感词：**<font color="warning">{DFA_result}</font>')
            flag = 1


        # 如果发现了敏感词，则跳出循环
        if flag == 1:
            break
        if len(input_text) <= 60:
            # 对输入文本进行预处理，移除HTML标签和重复字符
            input_text = replace_word(input_text)
            # 对输入文本进行中文分词
            input_text = chinese_word_cut(input_text)
            # 移除输入文本中的停用词
            input_text = remove_stopwords(input_text)
            # 将输入文本转换为模型可以接受的序列
            new_short_sequences = tokenizer.texts_to_sequences([input_text])
            # 对序列进行填充，使其长度统一为60
            new_short_padded_seqs = pad_sequences(new_short_sequences, maxlen=60, padding='post', value=1e-10)

            # 使用模型对输入文本进行预测
            predictions = model.predict(new_short_padded_seqs)
            # 根据预测结果，判断文本是否为敏感内容
            prediction = (predictions[:, 1] >= 0.5).astype(int)
            print("预测的标签:", prediction)

            if prediction == 1:
                # 如果预测为敏感，则打印相关信息
                print(f"敏感 {predictions[:, 1]} id:{id - 1} ")
                print(md)
                mgw = []

                # 对输入文本进行分词，并对每个词进行预测
                new_word_sequences = tokenizer.texts_to_sequences(input_text)
                new_word_padded_seqs = pad_sequences(new_word_sequences, maxlen=60, padding='post', value=1e-10)
                word_predictions = model.predict(new_word_padded_seqs)

                for word, word_prediction in zip(input_text.split(), word_predictions):
                    if word_prediction[1] > 0.5:
                        mgw.append(word)

                print(mgw)
                # 发送Markdown消息，包含敏感度和敏感词
                send_md(webhook,
                        content=f'**帖子内容：**<font color="warning">{md}</font>\n**敏感度：**<font color="warning">{predictions[:, 1]}（敏感）</font>\n**敏感词：**<font color="warning">{mgw}</font>')
            else:
                # 如果预测为正常，则发送Markdown消息
                send_md(webhook,
                        content=f'**帖子内容：**<font color="info">{md}</font>\n**敏感度：**<font color="info">{predictions[:, 1]}（正常）</font>')
                print(predictions[:, 1])
                print("正常")
                print(md)
            print()
            break

        elif len(input_text) >= 60:
            # 如果输入文本长度大于等于60，则对长文本进行处理
            print(input_text)
            # 替换文本中的HTML标签和重复字符
            input_text = replace_word(input_text)
            # 对长文本进行平滑分割
            input_long_text_smooth_split = list(map(smooth_split, [input_text]))[0]

            print("smooth_split:", input_long_text_smooth_split)

            # 对平滑分割后的长文本进行中文分词
            input_long_text = list(map(chinese_word_cut_long, input_long_text_smooth_split))
            # 对长文本进行停用词移除
            input_long_text = list(map(remove_stopwords_long, input_long_text))
            print(input_long_text)

            # 将长文本转换为模型可以接受的序列
            new_long_sequences = tokenizer.texts_to_sequences(input_long_text)
            # 对序列进行填充，使其长度统一为60
            new_long_padded_seqs = pad_sequences(new_long_sequences, maxlen=60, padding='post', value=1e-10)

            # 使用模型对长文本进行预测
            predictions = model.predict(new_long_padded_seqs)
            # 获取预测标签
            predicted_labels = np.argmax(predictions, axis=1)
            # 初始化敏感详情列表
            sensitive_details = []
            # 检查预测结果中是否有敏感度大于等于0.5的文本
            if np.any(predictions[:, 1] >= 0.5):
                # 遍历平滑分割后的文本、原始文本和预测结果
                for text_smooth_split, text, prediction in zip(input_long_text_smooth_split, input_long_text,
                                                               predictions):
                    # 如果当前文本的敏感度大于等于0.5，则进行进一步处理
                    if prediction[1] >= 0.5:
                        mgw = []  # 初始化敏感词列表
                        # 将文本转换为模型可以接受的序列
                        new_word_sequences = tokenizer.texts_to_sequences(text)
                        # 对序列进行填充，使其长度统一为60
                        new_word_padded_seqs = pad_sequences(new_word_sequences, maxlen=60, padding='post', value=1e-10)
                        # 使用模型对文本进行预测
                        word_predictions = model.predict(new_word_padded_seqs)

                        # 遍历文本和预测结果，查找敏感词
                        for word, word_prediction in zip(text, word_predictions):
                            if word_prediction[1] > 0.5:
                                mgw.append((word, word_prediction[1]))  # 将敏感词和其敏感度添加到列表中

                        # 将平滑分割后的文本、敏感词列表和敏感度添加到敏感详情列表中
                        sensitive_details.append((text_smooth_split, mgw, prediction[1]))

                        formatted_sensitive_details = []  # 初始化格式化后的敏感详情列表
                        # 遍历敏感详情列表，格式化每条敏感详情
                        for sentence, words, sentence_prediction in sensitive_details:
                            sentence_details = [f"{word} ({prob:.2f})" for word, prob in words]  # 格式化敏感词和其敏感度
                            formatted_sentence = f'"{sentence}": 句子敏感度 {sentence_prediction:.2f}, 包含敏感词: {", ".join(sentence_details)}'  # 格式化整个句子的敏感详情
                            formatted_sensitive_details.append(formatted_sentence)  # 将格式化后的敏感详情添加到列表中

                        # 将格式化的字符串列表连接成一个单一的字符串，用于显示
                        sensitive_details_str = '\n'.join(formatted_sensitive_details)

                # 发送Markdown格式的消息，包括帖子内容和敏感详情
                send_md(webhook,
                        content=f'**帖子内容：**<font color="warning">{md}</font>\n\n'
                                f'**敏感详情：**\n{sensitive_details_str}')
            else:
                # 如果预测为正常，则发送Markdown消息
                send_md(webhook,
                        content=f'**帖子内容：**<font color="info">{md}</font>\n**敏感度：**<font color="info">{predictions[:, 1]}（正常）</font>')
                print(predictions[:, 1])
                print("正常")
                print(md)

            break
