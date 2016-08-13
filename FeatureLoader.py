import numpy as np
import json

dict= json.load(open("DataSets/dataset50transpose.json"))


def get_training_data():
    infile = open("DataSets/trainingdata.txt", "r")
    y = np.asmatrix(np.zeros(8)).reshape(1, 8)
    x = np.asmatrix(np.zeros(len(dict))).reshape(1, len(dict))
    X = x
    Y = y
    i = 0

    for line in infile:
        if i != 0:
            words = line.split(" ")
            classified = int(words[0])
            X = np.vstack((X, x))
            for word in words[1:]:
                if word in dict:
                    X[i, int(dict[word]["position"])] = 1
            Y = np.vstack((Y, y))
            Y[i, classified-1] = 1
        i += 1

    infile.close()
    X = X[1:, :]
    Y = Y[1:, :]
    X = np.hstack((np.ones(X.shape[0]).reshape(X.shape[0], 1), X))
    return X, Y


def get_testing_data(data):
    y = np.asmatrix(np.zeros(8)).reshape(1, 8)
    x = np.asmatrix(np.zeros(len(dict))).reshape(1, len(dict))
    X = x
    Y = y

    words = data.split(" ")
    for word in words:
        if word in dict:
            X[0, int(dict[word]["position"])] = 1

    X = np.hstack((np.ones(X.shape[0]).reshape(X.shape[0], 1), X))
    return X