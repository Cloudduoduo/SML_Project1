import json
from collections import Counter

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


'''查看数据集是否balance'''
train_data_lable1 = [label["label"] for label in train_data1]
train_data_lable2 = [label["label"] for label in train_data2]

print(Counter(train_data_lable1))  # Counter({1: 2500, 0: 2500})
print(Counter(train_data_lable2))  # Counter({0: 11500, 1: 1500})  unbalanced
