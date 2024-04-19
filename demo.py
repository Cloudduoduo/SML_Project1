import json
from collections import Counter
import seaborn
import matplotlib.pyplot as plt

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

count_train_data1 = [len(doc["text"]) for doc in train_data1]
count_train_data2 = [len(doc["text"]) for doc in train_data2]
# print(count_train_data1)
seaborn.histplot(count_train_data1)
plt.show()
seaborn.histplot(count_train_data2)
plt.show()

'''bag of word + traditional ML'''

