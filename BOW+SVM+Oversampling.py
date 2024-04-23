import json
from collections import Counter
import seaborn
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import vstack
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler  # Import the over-sampler

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

train_data_label1 = [label["label"] for label in train_data1]
train_data_label2 = [label["label"] for label in train_data2]



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

train_data1_doc = [doc["text"] for doc in train_data1]
train_data2_doc = [doc["text"] for doc in train_data2]
test_doc = [doc["text"] for doc in test_data]

vectorizer.fit(train_data1_doc)
vectorizer.fit(train_data2_doc)

train_data1_x = vectorizer.transform(train_data1_doc)
train_data2_x = vectorizer.transform(train_data2_doc)
real_test_x = vectorizer.transform(test_doc)

'''Apply Over Sampling'''
ros = RandomOverSampler(random_state=42)
X_train1, y_train1 = ros.fit_resample(train_data1_x, train_data_label1)
X_train2, y_train2 = ros.fit_resample(train_data2_x, train_data_label2)

'''Combine training data from both domains'''
X_train = vstack([X_train1, X_train2])
y_train = y_train1 + y_train2

'''SVM'''
svm = LinearSVC()
svm.fit(X_train, y_train)
validation_prediction = svm.predict(X_validation)

'''Evaluate model'''
auc = roc_auc_score(y_validation, validation_prediction)
print(f"auc = {auc}")

'''Prepare for real test set prediction'''
svm = LinearSVC()
svm.fit(X_train, y_train)
real_test_prediction = svm.predict(real_test_x)

'''Generate submission file'''
submission_id = [ids["id"] for ids in test_data]
with open("svm_prediction.csv", "w") as file:
    file.write("id,class\n")
    for id_, pred_ in zip(submission_id, real_test_prediction):
        file.write(f"{id_}, {pred_}\n")
