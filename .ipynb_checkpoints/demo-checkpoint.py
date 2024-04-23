import json
from collections import Counter
import seaborn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import vstack
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
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
    preprocessor=preprocessor
)
def augment_text(text, num_augmented_samples=6):
    augmented_texts = []
    pring(text)
    words = text.split()
    for _ in range(num_augmented_samples):
        if random.choice([True, False]):
            augmented_texts.append(' '.join(random_insertion(words)))
        else:
            augmented_texts.append(' '.join(random_deletion(words)))
        augmented_texts.append(synonym_replacement(text, n=3))
    return augmented_texts

train_data1_doc = [doc["text"] for doc in train_data1]
train_data2_doc = [doc["text"] for doc in train_data2]
test_doc = [doc["text"] for doc in test_data]

# # 不可以用两个不一样的vectorizer，因为会得到两个维度不一样的矩阵，并且词也会对不上。最后是要在两个domain train data 上做预测的。 所以构建的特征矩阵必须用同一个词典
vectorizer.fit(train_data1_doc)
vectorizer.fit(train_data2_doc)

# 假设 1 为少数类
minority_data = [data for data, label in zip(train_data2_doc, train_data_lable2) if label == 1]

augmented_data = []
augmented_labels = []
for text in minority_data:
    augmented_texts = augment_text(text, num_augmented_samples=3)  # 为每个少数类样本生成3个新样本
    augmented_data.extend(augmented_texts)
    augmented_labels.extend([1] * len(augmented_texts))

# 将增强后的数据添加到原数据集中
train_data2_doc.extend(augmented_data)
train_data_lable2.extend(augmented_labels)

# 重新向量化处理
train_data2_x = vectorizer.transform(train_data2_doc)




train_data1_x = vectorizer.transform(train_data1_doc)
# train_data2_x = vectorizer.transform(train_data2_doc)
real_test_x = vectorizer.transform(test_doc)

# print(train_data1_x.shape) (5000, 71481)
# print(train_data2_x.shape) (13000, 71481)

'''把数据分为训练集，验证集和测试集'''
X_train1, X_others1, y_train1, y_others1 = train_test_split(train_data1_x,
                                                            train_data_lable1, test_size=0.3,
                                                            stratify=train_data_lable1)
X_validation1, X_test1, y_validation1, y_test1 = train_test_split(X_others1, y_others1, test_size=0.5,
                                                                  stratify=y_others1)
print(X_train1.shape, X_validation1.shape, X_test1.shape)

X_train2, X_others2, y_train2, y_others2 = train_test_split(train_data2_x,
                                                            train_data_lable2, test_size=0.3,
                                                            stratify=train_data_lable2)
X_validation2, X_test2, y_validation2, y_test2 = train_test_split(X_others2, y_others2, test_size=0.5,
                                                                  stratify=y_others2)
print(X_train2.shape, X_validation2.shape, X_test2.shape)
# stratify=y_others2 是因为train data中的label严重不平衡，可能导致拆分后有的集label相差太多
# 初始化 RandomOverSampler 实例
# ros = RandomOverSampler(random_state=42)
#
# # 只对 train_data2 的训练数据应用过采样
# X_train2_resampled, y_train2_resampled = ros.fit_resample(X_train2, y_train2)
#
# # 合并来自 domain1 和经过过采样的 domain2 的训练数据
# X_train = vstack([X_train1, X_train2_resampled])
# y_train = y_train1 + y_train2_resampled

# 现在你的 X_train 和 y_train 都是处理过类别不平衡的数据了

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
print("auc = ", auc, sep="")

'''hyper-parameter search
frequency matrix, feature = 71481
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
roc_auc_score(y_test, my_test_prediction)

'''在真实测试集上测试'''

svm = LinearSVC(C=0.1)
svm.fit(X_train, y_train)
real_test_prediction = svm.predict(real_test_x)

submission_id = [ids["id"] for ids in test_data]
with open("svm_prediction.csv", "w") as file:
    file.write("id,class\n")
    for id_, pred_ in zip(real_test_prediction, real_test_prediction):
        file.write(f"{id_}, {pred_}\n")