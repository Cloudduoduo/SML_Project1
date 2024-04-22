import json
import random
from collections import Counter
import seaborn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import vstack
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfTransformer

''' 读取文件'''


def read_json(filename):
    res = []
    with open(filename, "r") as file:
        for line in file:
            res.append(json.loads(line))
    return res


train_data1 = read_json("dataset/domain1_train_data.json")
train_data2 = read_json("dataset/domain2_train_data.json")
test_data = read_json("dataset/test_data.json")

'''查看数据集是否balance 以及每个doc的文本长度'''
train_data_label1 = [label["label"] for label in train_data1]
train_data_label2 = [label["label"] for label in train_data2]

# print(Counter(train_data_label1))  # Counter({1: 2500, 0: 2500})
# print(Counter(train_data_label2))  # Counter({0: 11500, 1: 1500})  unbalanced

count_train_data1 = [len(doc["text"]) for doc in train_data1]
count_train_data2 = [len(doc["text"]) for doc in train_data2]

'''bag of word + traditional ML'''


def preprocessor(doc):
    res = [str(text) for text in doc]
    return res


def tokenizer(doc):
    return doc


vectorizer = CountVectorizer(
    tokenizer=tokenizer,
    preprocessor=preprocessor,
    ngram_range=(1, 2),
    max_features=5000,
)

train_data1_doc = [doc["text"] for doc in train_data1]
train_data2_doc = [doc["text"] for doc in train_data2]
test_doc = [doc["text"] for doc in test_data]

# 不可以用两个不一样的vectorizer，因为会得到两个维度不一样的矩阵，并且词也会对不上。最后是要在两个domain train data 上做预测的。 所以构建的特征矩阵必须用同一个词典
vectorizer.fit(train_data1_doc)
vectorizer.fit(train_data2_doc)

train_data1_x = vectorizer.transform(train_data1_doc)
train_data2_x = vectorizer.transform(train_data2_doc)
real_test_x = vectorizer.transform(test_doc)

# TF-IDF
tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(train_data1_x)
tfidf_transformer.fit(train_data2_x)

train_data1_x = tfidf_transformer.transform(train_data1_x)
train_data2_x = tfidf_transformer.transform(train_data2_x)
real_test_x = tfidf_transformer.transform(real_test_x)

# print(train_data1_x.shape)  # (5000, 5000)
# print(train_data2_x.shape)  # (13000, 5000)

'''把数据分为训练集，验证集和测试集'''
X_train1, X_others1, y_train1, y_others1 = train_test_split(train_data1_x,
                                                            train_data_label1, test_size=0.3,
                                                            stratify=train_data_label1)
X_validation1, X_test1, y_validation1, y_test1 = train_test_split(X_others1, y_others1, test_size=0.5,
                                                                  stratify=y_others1)
# print(X_train1.shape, X_validation1.shape, X_test1.shape)

X_train2, X_others2, y_train2, y_others2 = train_test_split(train_data2_x,
                                                            train_data_label2, test_size=0.3,
                                                            stratify=train_data_label2)
X_validation2, X_test2, y_validation2, y_test2 = train_test_split(X_others2, y_others2, test_size=0.5,
                                                                  stratify=y_others2)
# print(X_train2.shape, X_validation2.shape, X_test2.shape)
# stratify=y_others2 是因为train data中的label严重不平衡，可能导致拆分后有的集label相差太多

X_train = vstack([X_train1, X_train2])
X_validation = vstack([X_validation1, X_validation2])
X_test = vstack([X_test1, X_test2])

y_train = y_train1 + y_train2
y_validation = y_validation1 + y_validation2
y_test = y_test1 + y_test2

'''SVM'''

svm = LinearSVC()
svm.fit(X_train, y_train)
validationPrediction = svm.predict(X_validation)

# 不用accuracy去评估模型。因为数据是不平衡的，所以accuracy无法评估模型。
auc = roc_auc_score(y_validation, validationPrediction)
# print("auc = ", auc, sep="")

# ----------------------------------------------------------------------------------------------------------------------


import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm


# 安装torch：去pytorch官网

class TokenizedTextDataset(Dataset):
    def __init__(self, tokenized_texts, labels, max_len=500):
        self.tokenized_texts = tokenized_texts
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.tokenized_texts)

    def __getitem__(self, idx):
        tokenized_text = self.tokenized_texts[idx][:self.max_len]
        label = self.labels[idx]

        return {
            "input_ids": torch.tensor(tokenized_text, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32)
        }


# 取到原始文本
text_domain1 = [x["text"] for x in train_data1]
text_domain2 = [x["text"] for x in train_data2]
y_domain1 = [x["label"] for x in train_data1]
y_domain2 = [x["label"] for x in train_data2]

# 可以在这个地方加data augmentation


text_test = [x["text"] for x in test_data]

dataset_domain1 = TokenizedTextDataset(text_domain1, y_domain1)
dataset_domain2 = TokenizedTextDataset(text_domain2, y_domain2)


def split_dataset(dataset, test_size=0.15, dev_size=0.15):
    labels = [dataset[idx]['label'] for idx in range(len(dataset))]

    positive_indexes = [index for index, label in enumerate(labels) if label == 1]
    negative_indexes = [index for index, label in enumerate(labels) if label == 0]

    pos_train, pos_other = train_test_split(positive_indexes, test_size=test_size + dev_size)
    pos_test, pos_dev = train_test_split(pos_other, test_size=0.5)

    neg_train, neg_other = train_test_split(negative_indexes, test_size=test_size + dev_size)
    neg_test, neg_dev = train_test_split(neg_other, test_size=0.5)

    train_indices = pos_train + neg_train
    dev_indices = pos_dev + neg_dev
    test_indices = pos_test + neg_test

    random.shuffle(train_indices)
    random.shuffle(dev_indices)
    random.shuffle(test_indices)

    train_set = [dataset[i] for i in train_indices]
    dev_set = [dataset[i] for i in dev_indices]
    test_set = [dataset[i] for i in test_indices]

    return train_set, dev_set, test_set


# 假设 dataset_domain1 和 dataset_domain2 已正确创建
d1_train, d1_dev, d1_test = split_dataset(dataset_domain1)
d2_train, d2_dev, d2_test = split_dataset(dataset_domain2)

print(len(d1_train), len(d1_dev), len(d1_test))
print(len(d2_train), len(d2_dev), len(d2_test))
