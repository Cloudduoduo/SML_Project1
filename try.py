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
    # labels = [dataset[idx]['label'] for idx in range(len(dataset))]
    labels = [idx['label'] for idx in dataset]

    positive_indexes = [index for index, label in enumerate(labels) if label == 1]
    negative_indexes = [index for index, label in enumerate(labels) if label == 0]

    pos_train, pos_other = train_test_split(positive_indexes, test_size=test_size + dev_size)
    pos_test, pos_dev = train_test_split(pos_other, test_size=0.5)

    neg_train, neg_other = train_test_split(negative_indexes, test_size=test_size + dev_size)
    neg_test, neg_dev = train_test_split(neg_other, test_size=0.5)

    train_indices = pos_train + neg_train
    dev_indices = pos_dev + neg_dev
    test_indices = pos_test + neg_test

    np.random.shuffle(train_indices)
    np.random.shuffle(dev_indices)
    np.random.shuffle(test_indices)

    train_set = [dataset[i] for i in train_indices]
    dev_set = [dataset[i] for i in dev_indices]
    test_set = [dataset[i] for i in test_indices]

    return train_set, dev_set, test_set


# 假设 dataset_domain1 和 dataset_domain2 已正确创建
d1_train, d1_dev, d1_test = split_dataset(dataset_domain1)
d2_train, d2_dev, d2_test = split_dataset(dataset_domain2)

print(len(d1_train), len(d1_dev), len(d1_test))
print(len(d2_train), len(d2_dev), len(d2_test))

train_data = d1_train + d2_train
random.shuffle(train_data)

dev_data = d1_dev + d2_dev
random.shuffle(dev_data)

test_data = d1_test + d2_test
random.shuffle(test_data)


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


# 训练模型

# create a model

# LSTM
vocab_size = 83582  # 词汇表的大小 自己算一下

# tunable hyper-parameter 这些超参数可以更改
embedding_dim = 50
hidden_dim = 128
num_layers = 1

# batch size
# learning rate (lr) [to be added]

output_dim = 1  # Binary classification 这个不变
model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers)

# initialize trainer

trainer = pl.Trainer(
    max_epochs=5,
    accelerator="auto"
)

trainer.fit(model, data_module)

# 在测试集上预测

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
