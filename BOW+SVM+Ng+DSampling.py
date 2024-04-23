"""预处理中修改ngram以及features的数量"""

import json
from collections import Counter
import seaborn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from scipy.sparse import vstack
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
import random
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

negative_train_data2 = []
for doc in train_data2:
    if doc["label"] == 0:
        negative_train_data2.append(doc)
positive_train_data2 = []
for doc in train_data2:
    if doc["label"] == 1:
        positive_train_data2.append(doc)

negative_train_data2 = random.sample(negative_train_data2, 1500)

train_data2 = negative_train_data2 + positive_train_data2


'''查看数据集是否balance 以及每个doc的文本长度'''
train_data_label1 = [label["label"] for label in train_data1]
train_data_label2 = [label["label"] for label in train_data2]

# print(Counter(train_data_lable1))  # Counter({1: 2500, 0: 2500})
# print(Counter(train_data_lable2))  # Counter({0: 11500, 1: 1500})  unbalanced


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




# print(train_data1_x.shape) # (5000, 5000)
# print(train_data2_x.shape) # (13000, 5000)

'''把数据分为训练集，验证集和测试集'''
X_train1, X_others1, y_train1, y_others1 = train_test_split(train_data1_x,
                                                            train_data_label1, test_size=0.3,
                                                            stratify=train_data_label1)
X_validation1, X_test1, y_validation1, y_test1 = train_test_split(X_others1, y_others1, test_size=0.5,
                                                                  stratify=y_others1)
print(X_train1.shape, X_validation1.shape, X_test1.shape)

X_train2, X_others2, y_train2, y_others2 = train_test_split(train_data2_x,
                                                            train_data_label2, test_size=0.3,
                                                            stratify=train_data_label2)
X_validation2, X_test2, y_validation2, y_test2 = train_test_split(X_others2, y_others2, test_size=0.5,
                                                                  stratify=y_others2)
print(X_train2.shape, X_validation2.shape, X_test2.shape)
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
validation_auc = roc_auc_score(y_validation, validationPrediction)
print("validation auc = ", validation_auc, sep="")

'''hyper-parameter search
frequency matrix, feature = 5000
'''

# for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
#     svm = LinearSVC(C=c, dual=True)
#     svm.fit(X_train, y_train)
#     validationPrediction = svm.predict(X_validation)
#     auc = roc_auc_score(y_validation, validationPrediction)
#     print(f"C = {c}, Auc = {auc}")
# c = 0.1 是最好的

svm = LinearSVC(C=0.1)
svm.fit(X_train, y_train)
my_test_prediction = svm.predict(X_test)
test_auc = roc_auc_score(y_test, my_test_prediction)
print("test auc = ", test_auc, sep="")
'''在真实测试集上测试'''

svm = LinearSVC(C=0.1)
svm.fit(X_train, y_train)
real_test_prediction = svm.predict(real_test_x)

submission_id = [ids["id"] for ids in test_data]
with open("svm_prediction+ng+DS+tf.csv", "w") as file:
    file.write("id,class\n")
    for id_, pred_ in zip(submission_id, real_test_prediction):
        file.write(f"{id_}, {pred_}\n")
