import pandas as pd
import torch
import numpy as np
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from Encoder import Encoder
from process_data import remove_chars

from Encoder import OneDataset


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return 'cpu'


def preprocess_data(data, scaler=None):
    # 确保输入是 PyTorch 张量
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)

    data = data.cpu()
    print(data.shape)
    # 如果没有提供 scaler，创建一个新的
    if scaler is None:
        scaler = StandardScaler()
        data_numpy = scaler.fit_transform(data.detach().numpy())
    else:
        data_numpy = scaler.transform(data.detach().numpy())

    # 将 numpy 数组转回 PyTorch 张量
    data_tensor = torch.FloatTensor(data_numpy).to(try_gpu())

    dump(scaler, 'scaler.joblib')
    return data_tensor


class PredictDataset(Dataset):
    def __init__(self, embeddings):
        super(PredictDataset, self).__init__()
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]


def predict(embeddings, encoder, classifier, args):
    raw = pd.read_csv(args.data_path)
    embeddings = np.load(embeddings)[:, -768:]

    scaler = load("./scaler.joblib")
    embeddings = preprocess_data(embeddings, scaler)

    dataloader = DataLoader(PredictDataset(embeddings), batch_size=10, shuffle=False)
    labels = []
    for embeddings in dataloader:
        classifier.eval()
        with torch.no_grad():
            res = classifier(embeddings)
            res = F.sigmoid(res)
            labels = labels + list(res.cpu().detach())
    labels = np.array(labels).reshape(-1)
    sentences = raw['example'].tolist()
    words = raw['word'].tolist()
    print(len(words), len(labels), len(sentences))
    output = pd.DataFrame({'word': words, 'example': sentences, 'label': labels})
    output.to_csv(args.model + f'out.csv', index=False)


def web_predict(word, sentence, classifier):
    encoder = Encoder()
    encoder.to(try_gpu())
    sentence = remove_chars(sentence)
    embeddings = encoder.encode(sentence, word)[:, -768:]

    scaler = load("./scaler.joblib")
    embeddings = preprocess_data(embeddings, scaler)

    labels = []
    classifier.eval()
    with torch.no_grad():
        res = classifier(embeddings)
        res = F.sigmoid(res)
        labels = labels + list(res.cpu().detach())
    labels = np.array(labels).reshape(-1)
    return labels[0]
