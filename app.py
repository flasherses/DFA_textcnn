from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model

import joblib
from DFA import DFAFilter
from predict import predict
app = Flask(__name__)
# 加载模型
model = load_model("model/fasttext_textcnn.h5")
# 加载分词器
tokenizer = joblib.load('model/tokenizer.pkl')

gfw = DFAFilter()
path="data/processed_data_content.csv"
gfw.parse(path)

@app.route('/predict', methods=['GET', 'POST'])
def predict_text():
    # 获取当前参数字典
    if request.method == 'POST':
        args = request.get_json()
        text = args.get('text', '')
    else:
        args = request.args
        text = args.get('text', '')
    # print(text)
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        result = predict(text,model,tokenizer,gfw)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 获取所有可用的GPU设备
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # 设置GPU内存自增长
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)
    app.run(host='0.0.0.0', port=5000, debug=True)
