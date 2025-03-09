import json
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

import re


def remove_chars(text):
    tag_pattern = r'#\w+#'
    text = re.sub(tag_pattern, '', text)


    # 定义正则表达式模式，匹配所有标点符号
    punctuation_pattern = r'[.,;!?！？，；：]'

    # 使用 re.sub() 函数将所有标点符号替换为逗号
    text = re.sub(punctuation_pattern, ',', text)

    # 定义正则表达式模式，匹配中文字符、英文字符和数字
    pattern = r'[^a-zA-Z0-9\u4e00-\u9fff,]'

    # 使用 re.sub() 函数替换不符合模式的字符
    cleaned_text = re.sub(pattern, '', text)

    if cleaned_text == '':
        cleaned_text = 'a'

    return cleaned_text


def to_csv():
    with open('filtered.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    for idx, row in df.iterrows():
        df.loc[idx, 'sentence'] = remove_chars(df.loc[idx, 'sentence'])
        if row['quality'] == 'good':
            df.loc[idx, 'quality'] = 1
        elif row['quality'] == 'bad':
            df.loc[idx, 'quality'] = 0

    df = df.rename(columns={'quality': 'label'})

    df.to_csv('filtered.csv', index=False, encoding='utf-8')
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv('train.csv', index=False, encoding='utf-8')
    test.to_csv('test.csv', index=False, encoding='utf-8')

if __name__ == '__main__':
    to_csv()



