import pickle

import numpy as np
import json

import pandas as pd

from Encoder import Encoder
import torch
from data_cheating_filtered import get_data
from process_data import remove_chars
import matplotlib.pyplot as plt


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return 'cpu'


def train_embedding():
    with open('./filtered.json', 'r', encoding='utf-8') as f:
        examples = json.load(f)

    encoder = Encoder()
    encoder.to(try_gpu())
    labels = []

    embeddings = None

    for i in range(16, len(examples), 16):
        sentences = []
        words = []
        for idx in range(i - 16, i):
            if examples[idx]['drop'] == 1:
                continue
            sentence = remove_chars(examples[idx]['sentence'])
            sentences.append(sentence)
            words.append(examples[idx]['word'])

            if examples[idx]['quality'] == "bad":
                labels.append(0)
            else:
                labels.append(1)

        with torch.no_grad():
            part_embedding = encoder.encode(sentences, words)
        print(part_embedding.device)
        print(part_embedding.shape)

        if embeddings is None:
            embeddings = part_embedding
        else:
            embeddings = torch.cat((embeddings, part_embedding), dim=0)

        print(embeddings.shape)

    labels = np.array(labels)
    np.save("label_new.npy", labels)
    print(embeddings.shape)
    embeddings = embeddings.cpu().numpy()
    np.save('embeddings_new.npy', embeddings)


def filter_false():
    with open('./filtered.json', 'r', encoding='utf-8') as f:
        examples = json.load(f)
    for i in range(len(examples)):
        sentence = examples[i]['sentence']
        drop = examples[i]['drop']
        if drop == 1:
            print(sentence)
            check = int(input())
            if check == 0:
                examples[i]['drop'] = 0

    with open('./filtered.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=4)


def arrange_buzzword_examples():
    raw = get_data()
    examples = []
    for key, value in raw.items():
        word = key
        sentences = value['examples']
        for sentence in sentences:
            examples.append({'word': word, 'sentence': sentence})
    with open('./buzzword.json', 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=4)


def save_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    words = []
    examples = []

    for item in data:
        words.append(item['word'])
        examples.append(item['sentence'])

    res = pd.DataFrame({'word': words, 'example': examples})
    res.to_csv('buzzword_examples.csv', index=False, encoding='utf-8')


def buzzword_embeddings():
    encoder = Encoder()
    encoder.to(try_gpu())
    with open('./buzzword.json', 'r', encoding='utf-8') as f:
        examples = json.load(f)

    embeddings = None
    for i in range(10, len(examples) + 1, 10):
        sentences = []
        words = []
        for idx in range(i - 10, i):
            print(idx)
            sentence = examples[idx]['sentence']
            word = examples[idx]['word']
            sentence = remove_chars(sentence)
            word = remove_chars(word)

            pos = sentence.find(word)
            if len(sentence) < 508:
                pass
            elif pos + 254 > len(sentence):
                sentence = sentence[len(sentence) - 508:]
            elif pos - 254 < 0:
                sentence = sentence[:508]
            else:
                sentence = sentence[pos-254:pos+254]

            if word not in sentence:
                sentence = sentence + ', ' + word

            sentences.append(sentence)
            words.append(word)

        print(len(sentences), len(words))

        with torch.no_grad():
            part_embedding = encoder.encode(sentences, words)
        if embeddings is None:
            embeddings = part_embedding
        else:
            embeddings = torch.cat((embeddings, part_embedding), dim=0)

    embeddings = embeddings.cpu().numpy()
    np.save('embeddings_buzzword.npy', embeddings)


def select_best():
    way1 = pd.read_csv('./MLPout_1536.csv', encoding='utf-8')
    way2 = pd.read_csv('./MLPout.csv', encoding='utf-8')

    # 按word分组，选择每个word中label最高的10个example
    df_top = way1.groupby('word', group_keys=False).apply(lambda x: x.nlargest(10, 'label'))
    df_top.to_csv('top10_MLPout_1536.csv', index=False, encoding='utf-8')

    df_top = way2.groupby('word', group_keys=False).apply(lambda x: x.nlargest(10, 'label'))
    df_top.to_csv('top10_MLPout.csv', index=False, encoding='utf-8')

    print(df_top)

    # 可选：按word排序
    # df_top = df_top.sort_values(by='word')


def selected():
    df_selected = pd.read_csv('./top10_MLPout.csv', encoding='utf-8')

    # 按word分组
    groups = df_selected.groupby('word')

    # 构建结果字典
    result = {}
    for word, group in groups:
        examples = group['example'].tolist()
        result[word] = {'example': examples}

    # 转换为JSON字符串
    json_data = json.dumps(result, ensure_ascii=False, indent=4)
    print(json_data)

    # 保存到JSON文件
    with open('data_selected.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


def sta():
    with open('filtered.json', 'r', encoding='utf-8') as f:
        examples = json.load(f)
    good = 0
    bad = 0
    drop = 0
    for item in examples:
        if item['quality'] == 'good':
            good += 1
        elif item['quality'] == 'bad':
            bad += 1
        if item['drop'] == 1:
            drop += 1
            print(item['sentence'])
    print(good, bad, drop, bad - drop)


def test_embeddings():
    test_data = pd.read_excel('./test.xlsx')
    buzzwords = test_data['word'].tolist()
    examples = test_data['example'].tolist()
    labels = test_data['label'].tolist()


    encoder = Encoder()
    encoder.to(try_gpu())
    test_labels = []

    embeddings = None
    for i in range(10, len(examples) + 1, 10):
        sentences = []
        words = []
        for idx in range(i - 10, i):
            print(idx)
            sentence = examples[idx]
            word = buzzwords[idx]
            sentence = remove_chars(sentence)
            word = remove_chars(word)

            pos = sentence.find(word)
            if len(sentence) < 508:
                pass
            elif pos + 254 > len(sentence):
                sentence = sentence[len(sentence) - 508:]
            elif pos - 254 < 0:
                sentence = sentence[:508]
            else:
                sentence = sentence[pos-254:pos+254]

            if word not in sentence:
                sentence = sentence + ', ' + word

            sentences.append(sentence)
            words.append(word)
            test_labels.append(labels[idx])

        print(len(sentences), len(words))

        with torch.no_grad():
            part_embedding = encoder.encode(sentences, words)
        if embeddings is None:
            embeddings = part_embedding
        else:
            embeddings = torch.cat((embeddings, part_embedding), dim=0)

    embeddings = embeddings.cpu().numpy()
    test_labels = np.array(test_labels)
    np.save('embeddings_test.npy', embeddings)
    np.save('labels_test.npy', test_labels)


def count_distribution():
    data = get_data()
    words = data.keys()
    distribution = {}
    length = []
    contamination = {}
    for word in words:
        sentences = data[word]['examples']
        distribution[word] = len(sentences)
        idx = int(len(sentences) / 10)
        length.append(len(sentences))
        from key_evaluation_results import gpt_4o_mini
        contamination_ = 0 if gpt_4o_mini[word]["准确性"][0] > 2 or \
                                                      gpt_4o_mini[word]["细节完整性"][0] > 2 else 1
        contamination[word] = contamination_

    df = pd.read_csv('example_score.csv')
    df_top = df.groupby('word', group_keys=False).apply(lambda x: x.nlargest(50, 'label'))

    for key, value in distribution.items():
        if value >= 50 and contamination[key] == 1:
            print(key, value, contamination[key])
            for index, row in df_top.iterrows():
                if row['word'] == key:
                    print(row['example'])


    length = np.array(length)
    # 设置直方图参数
    bin_width = 10
    min_val = np.floor(length.min())
    max_val = np.ceil(length.max())

    # 计算bins的边界
    bins = np.arange(min_val, max_val + bin_width, bin_width)

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(length, bins=bins, edgecolor='black', alpha=0.7)
    plt.title('Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def identify_reasons():
    ran1 = []
    ran2 = []
    waus1 = []
    waus2 = []
    waus = pickle.load(open('./data/TIGRESS_old_gpt-4o-mini_sentence_waus10_critic_nums_3_critic_ratio_0.5_w_evaluation.pickle', 'rb'))
    random_ = pickle.load(open('./data/TIGRESS_old_gpt-4o-mini_sentence_random10_critic_nums_3_critic_ratio_0.5_w_evaluation.pickle', 'rb'))
    print(waus['白人饭'])
    data = get_data()

    for key, value in random_.items():
        waus_sa = waus[key]['evaluation']['准确性'][0]
        random_sa = random_[key]['evaluation']['准确性'][0]
        waus_len = len(data[key]['examples'])
        random_len = len(random_[key]['examples'])


        # if random_sa == 1:
        #     print('word: ', key, 'random score: ', random_sa, 'waus_score: ', waus_sa)
        #     print('waus_res: ', waus[key]['predicted_definition'], '\n',
        #           'waus_reasons: ', waus[key]['evaluation']['准确性'][1:], '\n'
        #           'ground_truth: ', random_[key]['ground_truth'])

        if random_sa > waus_sa:
            # print('word: ', key, 'random score: ', random_sa, 'waus_score: ', waus_sa)
            # print('random_res: ', random_[key]['predicted_definition'], '\n',
            #       'random_reasons: ', random_[key]['evaluation']['准确性'][1:], '\n'
            #       'waus_res: ', waus[key]['predicted_definition'], '\n',
            #       'waus_reasons: ', waus[key]['evaluation']['准确性'][1:], '\n'
            #       'ground_truth: ', random_[key]['ground_truth'])
            ran1.append(random_len)
            waus1.append(waus_len)
        elif random_sa < waus_sa:
            ran2.append(random_len)
            waus2.append(waus_len)
    print('random > waus: ', np.array(ran1).mean(), np.array(waus1).mean())
    print('random < waus: ', np.array(ran2).mean(), np.array(waus2).mean())


if __name__ == '__main__':
    # train_embedding()
    # test_embeddings()
    # count_distribution()
    identify_reasons()

