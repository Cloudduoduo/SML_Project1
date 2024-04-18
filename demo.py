import json


def read_json(filename):
    res = []
    with open(filename, "r") as file:
        for line in file:
            res.append(json.loads(line))
    return res


domain_train1 = read_json("dataset/domain1_train_data.json")
domain_train2 = read_json("dataset/domain2_train_data.json")