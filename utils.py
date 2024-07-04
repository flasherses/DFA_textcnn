import re
import jieba

def chinese_word_cut(text):
    # print(text)
    # 移除文本中的图片标签
    text = re.sub(r'<img.*?>', '', text)
    # 移除文本中的div标签
    text = re.sub(r'<div.*?</div>', '', text)
    # 移除文本中的p标签
    text = re.sub(r'<p>', '', text)
    text = re.sub(r'</p>', '', text)

    # 移除文本中的英文句号
    text = text.replace('.', '')
    # 移除文本中的英文逗号
    text = text.replace(',', '')
    # 移除文本中的中文逗号
    text = text.replace('，', '')
    # 移除文本中的连续重复字符
    text = re.sub(r'(.)\1{3,}', r'\1', text)
    # 如果文本长度小于4，则直接返回文本
    if len(text) < 4:
        return text
    # 加载自定义词典
    jieba.load_userdict('data/custom_dict.txt')
    # 对文本进行分词，并用空格连接
    return " ".join(jieba.cut(text))

def remove_stopwords(text):
    stopwords = set()
    # 打开停用词文件
    with open('data/chinese_stopwords.txt', 'r', encoding='utf-8') as f:
        # 读取停用词并添加到集合中
        for line in f:
            stopwords.add(line.strip())
    # 去除文本中的停用词
    words = [word for word in text.split() if word not in stopwords]
    # 将去除停用词后的词语用空格连接
    return " ".join(words)
    # return ''.join(lazy_pinyin(" ".join(words)))


    # 替换文本中的HTML标签和重复字符
def replace_word(text):
    # 移除文本中的p标签
    text = re.sub(r'<p>', '', text)
    # 移除文本中的p标签
    text = re.sub(r'</p>', '', text)
    # 移除文本中的img标签
    text = re.sub(r'<img.*?>', '', text)
    # 移除文本中的div标签
    text = re.sub(r'<div.*?</div>', '', text)
    text = re.sub(r'&nbsp','',text)
    return text



# 对长文本进行分词处理
def chinese_word_cut_long(text):
    # 移除文本中的img标签
    text = re.sub(r'<img.*?>', '', text)
    # 移除文本中的div标签
    text = re.sub(r'<div.*?</div>', '', text)
    # 移除文本中的英文句号
    text = text.replace('.', '')
    # 移除文本中的英文逗号
    text = text.replace(',', '')
    # 移除文本中的中文逗号
    text = text.replace('，', '')
    # 移除文本中的连续重复字符
    text = re.sub(r'(.)\1{3,}', r'\1', text)
    # 如果文本长度小于4，则直接返回文本
    if len(text) < 4:
        return text
    # 加载自定义词典
    jieba.load_userdict('data/custom_dict.txt')
    # 对文本进行分词，并用空格连接
    text = " ".join(jieba.cut(text))
    return text


# 移除长文本中的停用词
def remove_stopwords_long(text):
    stopwords = set()
    # 打开停用词文件
    with open('data/chinese_stopwords.txt', 'r', encoding='utf-8') as f:
        # 读取停用词并添加到集合中
        for line in f:
            stopwords.add(line.strip())
    # 去除文本中的停用词
    words = [word for word in text.split() if word not in stopwords]
    return words


# 平滑分割长文本
def smooth_split(text):
    max_length=60
    step=50
    sub_texts = []
    for i in range(0, len(text), step):  
        sub_texts.append(text[i:i + max_length]) 
    return sub_texts