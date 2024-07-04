from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import numpy as np
import jieba
import re
from utils import chinese_word_cut, remove_stopwords, replace_word, chinese_word_cut_long, remove_stopwords_long, \
    smooth_split
from DFA import DFAFilter
import time

# 定义预测函数
def predict(input_text,model,tokenizer,gfw):
    # 打印当前系统时间
    print("当前系统时间:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    result = {}
    DFA_result = gfw.filter(input_text)
    if DFA_result:
        result['文本'] = [input_text]
        result['敏感词'] = DFA_result
        result['敏感度'] = 1
        result['function'] = 'DFA'
        print("DFA_result:", result)
        return result

    
    text = input_text
    input_text = re.sub(r'\d+', '', input_text)
    input_text = input_text.replace(' ', '')
    input_text = replace_word(input_text)
    if len(input_text) <= 60:
        print("短文本")

        
        print("正则后text:",input_text)
        input_text = chinese_word_cut(input_text)
        input_text = remove_stopwords(input_text)

        # 将输入文本转换为序列
        new_sequences = tokenizer.texts_to_sequences([input_text])
        print("new_sequences:", new_sequences)
        # 对序列进行填充，确保长度为60，填充值为1e-10
        new_padded_seqs = pad_sequences(new_sequences, maxlen=60, padding='post', value=1e-10)
        # 使用模型进行预测
        predictions = model.predict(new_padded_seqs)
        print("模型预测返回:", predictions)
        # 根据预测结果，判断是否为敏感文本
        prediction = (predictions[:, 1] >= 0.5).astype(int)
        print("预测的标签:", prediction)

        if prediction == 1:
            print(f"敏感 {predictions[:, 1]}")
            mgw = []
            mgw_prob = []
            # 将输入文本转换为序列，用于敏感词检测
            new_word_sequences = tokenizer.texts_to_sequences(input_text)
            
            # 对序列进行填充，确保长度为60，填充值为1e-10
            new_word_padded_seqs = pad_sequences(new_word_sequences, maxlen=60, padding='post', value=1e-10)
            # 使用模型进行敏感词预测
            word_predictions = model.predict(new_word_padded_seqs)
            # print("敏感词预测结果:", word_predictions)

            for word, word_prediction in zip(input_text, word_predictions):
                print(word,word_prediction)
                # 判断是否为敏感词
                if word_prediction[1] > 0.5:
                    print(f"敏感词 {word}")
                    mgw.append(word)
                    mgw_prob.append(float(word_prediction[1]))

            # print(mgw)

            result['文本'] = [text]
            result['敏感度'] = predictions[:, 1].tolist()  # 将numpy数组转换为列表以便JSON序列化
            result['敏感词'] = mgw
            result['敏感词概率'] = mgw_prob
            result['预测的标签'] = prediction.tolist()  # 将numpy数组转换为列表以便JSON序列化
            result['function'] = 'textcnn'
            print(result)
            return result
        else:
            print(predictions[:, 1])
            result['文本'] = [text]
            result['敏感度'] = predictions[:, 1].tolist()  # 将numpy数组转换为列表以便JSON序列化
            result['预测的标签'] = prediction.tolist()  # 将numpy数组转换为列表以便JSON序列化
            result['function'] = 'textcnn'
            print(result)
            return result

    elif len(input_text) >= 60:
        print("长文本")
        # input_text = replace_word(input_text)
        input_long_text_smooth_split = list(map(smooth_split,[input_text]))[0]
        print("smooth_split:", input_long_text_smooth_split)

        input_long_text = list(map(chinese_word_cut_long, input_long_text_smooth_split))
        input_long_text = list(map(remove_stopwords_long, input_long_text))
        # print(input_long_text)

        new_long_sequences = tokenizer.texts_to_sequences(input_long_text)
        new_long_padded_seqs = pad_sequences(new_long_sequences, maxlen=60, padding='post', value=1e-10)

        predictions = model.predict(new_long_padded_seqs)
        print("模型预测结果:", predictions)
        prediction_label = (predictions[:, 1] >= 0.5).astype(int)
        print("预测的标签:", prediction_label)
        sensitive_details = []
        if np.any(predictions[:, 1] >= 0.5):
            for text_smooth_split, text, prediction in zip(input_long_text_smooth_split, input_long_text, predictions):
                if prediction[1] >= 0.5:
                    mgw = []
                    new_word_sequences = tokenizer.texts_to_sequences(text)
                    new_word_padded_seqs = pad_sequences(new_word_sequences, maxlen=60, padding='post', value=1e-10)
                    word_predictions = model.predict(new_word_padded_seqs)

                    for word, word_prediction in zip(text, word_predictions):
                        if word_prediction[1] > 0.5:
                            mgw.append((word, word_prediction[1]))

                    sensitive_details.append((text_smooth_split, mgw, prediction[1]))

                    formatted_sensitive_details = []
                    for sentence, words, sentence_prediction in sensitive_details:
                        sentence_details = [f"{word} ({prob:.2f})" for word, prob in words]
                        formatted_sentence = f'"{sentence}": 句子敏感度 {sentence_prediction:.2f}, 包含敏感词: {", ".join(sentence_details)}'
                        formatted_sensitive_details.append(formatted_sentence)

                    sensitive_details_str = '\n'.join(formatted_sensitive_details)
            result['文本'] = [text]
            result['敏感详情'] = sensitive_details_str
            result['预测的标签'] = prediction_label.tolist()  # 将numpy数组转换为列表以便JSON序列化
            result['function'] = 'textcnn'
            print(result)
            return result
        else:
            
            result['文本'] = [text]
            result['敏感度'] = predictions[:, 1].tolist()  # 将numpy数组转换为列表以便JSON序列化
            result['预测的标签'] = prediction_label.tolist()  # 将numpy数组转换为列表以便JSON序列化
            result['function'] = 'textcnn'
            print(result)
            return result