"""
sequence-model(rnn, lstm, gru)
"""
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

# print(train_data1[0])


'''查看数据集是否balance 以及每个doc的文本长度'''
train_data_label1 = [label["label"] for label in train_data1]
train_data_label2 = [label["label"] for label in train_data2]

# print(Counter(train_data_label1))  # Counter({1: 2500, 0: 2500})
# print(Counter(train_data_label2))  # Counter({0: 11500, 1: 1500})  unbalanced

count_train_data1 = [len(doc["text"]) for doc in train_data1]
count_train_data2 = [len(doc["text"]) for doc in train_data2]
# print(count_train_data1)
seaborn.histplot(count_train_data1)
# plt.show()
seaborn.histplot(count_train_data2)
# plt.show()

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

'''
hyper-parameter search
ngram (1,2)
frequency matrix, feature = 5000
'''

# for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
#     svm = LinearSVC(C=c, dual=True)
#     svm.fit(X_train, y_train)
#     validationPrediction = svm.predict(X_validation)
#     auc = roc_auc_score(y_validation, validationPrediction)
#     print(f"C = {c}, Auc = {auc}")
# c = 0.1 是最好的

# svm = LinearSVC(C=0.1)
# svm.fit(X_train, y_train)
# my_test_prediction = svm.predict(X_test)
# roc_auc_score(y_test, my_test_prediction)
#
# '''在真实测试集上测试'''
#
# svm = LinearSVC(C=0.1)
# svm.fit(X_train, y_train)
# real_test_prediction = svm.predict(real_test_x)
#
# submission_id = [ids["id"] for ids in test_data]
# with open("svm_prediction.csv", "w") as file:
#     file.write("id,class\n")
#     for id_, pred_ in zip(real_test_prediction, real_test_prediction):
#         file.write(f"{id_}, {pred_}\n")


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
    # get labels from dataset
    labels = [example["label"].item() for example in dataset]

    # find the indices of positive and negative samples
    positive_indexes = [index for index, label in enumerate(labels) if label == 1]
    negative_indexes = [index for index, label in enumerate(labels) if label == 0]

    # split positive and negative samples separately
    pos_train, pos_other = train_test_split(positive_indexes, test_size=test_size + dev_size)
    pos_test, pos_dev = train_test_split(pos_other, test_size=0.5)

    neg_train, neg_other = train_test_split(negative_indexes, test_size=test_size + dev_size)
    neg_test, neg_dev = train_test_split(neg_other, test_size=0.5)

    # combine positive and negative samples to get train/dev/test splits
    train_indices = pos_train + neg_train
    dev_indices = pos_dev + neg_dev
    test_indices = pos_test + neg_test

    # shuffle the indices to randomize the other
    np.random.shuffle(train_indices)
    np.random.shuffle(dev_indices)
    np.random.shuffle(test_indices)

    # create subsets based on the indices
    train_set = [dataset[i] for i in train_indices]
    dev_set = [dataset[i] for i in dev_indices]
    test_set = [dataset[i] for i in test_indices]

    return train_set, dev_set, test_set


# split d1,d2 dataset
d1_train, d1_dev, d1_test = split_dataset(train_data1)
d2_train, d2_dev, d2_test = split_dataset(train_data2)

print(len(d1_train), len(d1_dev), len(d1_test))
print(len(d2_train), len(d2_dev), len(d2_test))

train_data = d1_train + d2_train
random.shuffle(train_data)

dev_data = d1_dev + d2_dev
random.shuffle(dev_data)

test_data = d1_test + d2_test
random.shuffle(test_data)


# date module可以自动每次把一个batch传给model， 进行反向传播计算梯度， 更新参数

class MyDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, collate_fn, batch_size=32):  # 16, 32, 64, 128
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn  # 定义了如何通过一个batch的数据构建x_matrix, y

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)


def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['label'] for item in batch]

    # Pad sequences to the maximum length in the batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,  # "square matrix"
        'label': torch.stack(labels)
    }


data_module = MyDataModule(train_data, dev_data, test_data, collate_fn, batch_size=16)


class LSTMClassifier(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Return the probability for the sequence
        embedded = self.embedding(x)  # Convert integer tokens to embeddings
        lstm_out, _ = self.lstm(embedded)  # Extract LSTM outputs
        lstm_out = lstm_out[:, -1, :]  # Select the last time-step output
        output = self.fc(lstm_out)  # Apply the linear layer
        output = self.sigmoid(output)  # Convert to probabilities
        return output


def training_step(self, batch, batch_idx):
    input_ids = batch['input_ids']  # x
    labels = batch['label']  # y
    outputs = self(input_ids)  # compute "sentence probabilities"
    loss = nn.BCELoss()(outputs.squeeze(), labels)  # compute loss
    self.log('train_loss', loss)  # save loss to class variable, for retrieval in other components
    return loss


def validation_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    labels = batch['label']
    outputs = self(input_ids)
    loss = nn.BCELoss()(outputs.squeeze(), labels)
    self.log('val_loss', loss)
    return loss


def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=0.001)


'''train model'''

# create a model

# LSTM
vocab_size = 83582  # 词汇表的大小

# tunable hyper-parameter
embedding_dim = 50
hidden_dim = 128
num_layers = 1

# batch size
# learning rate (lr) [to be added]

output_dim = 1  # Binary classification
model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers)

# initialize trainer

trainer = pl.Trainer(
    max_epochs=5,
    accelerator="auto"
)

trainer.fit(model, data_module)

model.eval()

# Lists to store true labels and predicted probabilities
true_labels = []
predicted_probs = []

# Iterate over the test set and make predictions
with torch.no_grad():
    for batch in tqdm(data_module.test_dataloader()):
        input_ids = batch['input_ids']
        labels = batch['label']
        outputs = model(input_ids)
        predicted_probs.extend(outputs.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

auc_score = roc_auc_score(true_labels, predicted_probs)
print("AUC Score:", auc_score)
