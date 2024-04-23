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

'''分割数据集'''
X_train1, X_others1, y_train1, y_others1 = train_test_split(train_data1_x,
                                                            train_data_label1, test_size=0.3,
                                                            stratify=train_data_label1)
X_validation1, X_test1, y_validation1, y_test1 = train_test_split(X_others1, y_others1, test_size=0.5,
                                                                  stratify=y_others1)

X_train2, X_others2, y_train2, y_others2 = train_test_split(train_data2_x,
                                                            train_data_label2, test_size=0.3,
                                                            stratify=train_data_label2)
X_validation2, X_test2, y_validation2, y_test2 = train_test_split(X_others2, y_others2, test_size=0.5,
                                                                  stratify=y_others2)

'''Apply Over Sampling'''
ros = RandomOverSampler(random_state=42)
X_train1, y_train1 = ros.fit_resample(X_train1, y_train1)
X_train2, y_train2 = ros.fit_resample(X_train2, y_train2)

'''Combine training data from both domains'''
X_train = vstack([X_train1, X_train2])
X_validation = vstack([X_validation1, X_validation2])
X_test = vstack([X_test1, X_test2])

y_train = y_train1 + y_train2
y_validation = y_validation1 + y_validation2
y_test = y_test1 + y_test2




'''SVM'''
svm = LinearSVC()
svm.fit(X_train, y_train)
validation_prediction = svm.predict(X_validation)

'''Evaluate model'''
val_auc = roc_auc_score(y_validation, validation_prediction)
print("Validation AUC:", val_auc)


svm.fit(X_train, y_train)
test_predictions = svm.predict(X_test)
test_auc = roc_auc_score(y_test, test_predictions)
print("Test AUC:", test_auc)

'''Prepare for real test set prediction'''
svm = LinearSVC()
svm.fit(X_train, y_train)
real_test_prediction = svm.predict(real_test_x)

'''Generate submission file'''
submission_id = [ids["id"] for ids in test_data]
with open("svm_prediction_oversampling.csv", "w") as file:
    file.write("id,class\n")
    for id_, pred_ in zip(submission_id, real_test_prediction):
        file.write(f"{id_}, {pred_}\n")
