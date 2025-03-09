import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
#from imblearn.under_sampling import RandomUnderSampler
from process_data import remove_chars

path = './filtered.csv'


def apply_mask(sentences, words):
    for idx, (sentence, word) in enumerate(zip(sentences, words)):
        print(sentence, word)
        sentences[idx] = sentence.replace(word, '[MASK]')
    return sentences


def get_pos(sentences, mask=103):
    pos = []
    for sentence in sentences:
        for idx, token in enumerate(sentence):
            if sentence[idx] == mask:
                pos.append(idx)
                break
    return pos


def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return 'cpu'


class OneDataset(Dataset):
    def __init__(self, data):
        super(OneDataset, self).__init__()
        self.words = []
        self.sentences = []
        for key, values in data.items():
            word = key
            sentences = values['examples']
            for sentence in sentences:
                if len(sentence) >= 512:
                    continue
                if word not in sentence:
                    continue
                self.words.append(remove_chars(word))
                self.sentences.append(remove_chars(sentence))
        #self.data = pd.read_csv(path)
        # X = self.data.drop(columns=['label'])
        # y = self.data['label']
        #
        # rus = RandomUnderSampler(random_state=27)
        #
        # self.X_resampled, self.y_resampled = rus.fit_resample(X, y)

    def __len__(self):
        return len(self.words)
        # return len(self.X_resampled)

    def __getitem__(self, idx):
        return self.words[idx], self.sentences[idx]
        # return self.X_resampled.iloc[idx]['word'], self.X_resampled.iloc[idx]['sentence'], self.y_resampled.iloc[idx]


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('../Chinese_BERT')
        self.encoder = BertModel.from_pretrained('../Chinese_BERT')

    def encode(self, sentences, words):
        sentences = apply_mask([sentences], [words])
        tokenized = self.tokenizer(text=sentences, padding=True)
        input_tokens = torch.tensor(tokenized['input_ids']).to(try_gpu())
        pos = get_pos(input_tokens)
        attention_mask = torch.tensor(tokenized['attention_mask']).to(try_gpu())
        embeddings = self.encoder(input_ids=input_tokens, attention_mask=attention_mask)
        embeddings = embeddings['last_hidden_state']
        cls_embeddings = embeddings[:, 0, :]
        masked_embeddings = embeddings[range(embeddings.shape[0]),  pos]
        return torch.cat((cls_embeddings, masked_embeddings), dim=-1)


# def get_embedding():
#     model = Encoder()
#     model.to(try_gpu())
#
#     data = get_data()
#
#     dataloader = DataLoader(OneDataset(data), batch_size=16, shuffle=True, drop_last=True)
#
#     emb = []
#     label_total = []
#     cnt = 0
#
#     dataframe = pd.DataFrame(columns=['examples'])
#     for word, sentence in dataloader:
#         for sen in sentence:
#             dataframe.loc[len(dataframe)] = [sen]
#
#         with torch.no_grad():
#             embeddings = model.encode(sentence, word)
#             emb.append(embeddings.cpu().numpy())
#             cnt += 16
#             print(cnt)
#
#     # for word, sentence, label in dataloader:
#     #     with torch.no_grad():
#     #         embeddings = model.encode(sentence, word)
#     #         emb.append(embeddings.cpu().numpy())
#     #         label_total.append(label.numpy())
#     #         cnt += 16
#     #         print(cnt)
#
#     print(cnt)
#
#     res = np.array(emb)
#     #label = np.array(label_total)
#     #print(label.shape)
#     #label = label.reshape(-1)
#     #res = res.reshape(label.shape[0], -1)
#
#     dataframe.to_csv('examples.csv', encoding='utf-8', index=False)
#
#     np.save('examples_embeddings.npy', res)
#     #np.save('label.npy', label)
#     #print(res.shape, label.shape)
#
