from flask import Flask, request, jsonify, render_template

from Classifiers import Classifier_MLP
from infer import web_predict
import torch

app = Flask(__name__)

def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return 'cpu'

@app.route('/process-input', methods=['POST'])
def process_input():
    data = request.get_json()
    word = data.get('word')
    sentence = data.get('sentence')
    print(word, sentence)

    # 调用你的模型进行计算，例如：
    result = float(my_model_function(word, sentence))

    return jsonify({'result': result})


def my_model_function(word, sentence):

    label = web_predict(word, sentence, classifier)
    return label

@app.route('/')
def home():
    return render_template('index.html')  # 假设你的HTML文件名为index.html


if __name__ == '__main__':
    classifier = Classifier_MLP(input_dim=768, dropout_rate=0)
    classifier.load_state_dict(torch.load('MLP' + '.pth'))
    classifier.to(try_gpu())
    app.run(debug=True)
