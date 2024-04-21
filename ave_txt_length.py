import json
from collections import Counter
import seaborn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import vstack
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score

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
train_data_lable1 = [label["label"] for label in train_data1]
train_data_lable2 = [label["label"] for label in train_data2]

# print(Counter(train_data_lable1))  # Counter({1: 2500, 0: 2500})
# print(Counter(train_data_lable2))  # Counter({0: 11500, 1: 1500})  unbalanced

domain1_countClass1 = 0
domain1_countClass0 = 0
for doc in train_data1:
    if doc["label"] == 1:
        domain1_countClass1 += len(doc["text"])
    elif doc["label"] == 0:
        domain1_countClass0 += len(doc["text"])
print(f"average text count in domain1 for class 1 is {domain1_countClass1/2500}")
print(f"average text count in domain1 for class 0 is {domain1_countClass0/2500}")

domain2_countClass1 = 0
domain2_countClass0 = 0
for doc in train_data2:
    if doc["label"] == 1:
        domain2_countClass1 += len(doc["text"])
    elif doc["label"] == 0:
        domain2_countClass0 += len(doc["text"])
print(f"average text count in domain1 for class 1 is {domain2_countClass1/1500}")
print(f"average text count in domain1 for class 0 is {domain2_countClass0/11500}")