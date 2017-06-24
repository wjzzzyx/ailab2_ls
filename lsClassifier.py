from __future__ import division
import numpy as np
from getFeature import X, y


def lsClassifier(traindata, trainlabel, testdata, testlabel, lamda):
    n = traindata.shape[0]
    m = traindata.shape[1]
    # calculate weights and b
    ones = np.ones((n, 1))
    X = np.column_stack((ones, traindata))
    X = np.mat(X)
    Z = (X.T.dot(X)) + lamda * np.eye(m + 1)
    Z = Z.I
    y = np.mat(trainlabel)
    y = y.T
    y1 = X.T.dot(y)
    w = Z.dot(y1)

    # test process
    n1 = testdata.shape[0]
    ones = np.ones((n1, 1))
    testdata = np.column_stack((ones, testdata))
    y_pred = testdata.dot(w)
    correct_count = 0
    for i in xrange(n1):
        if y_pred[i] > 0:
            y_pred[i] = 1
        else:
            y_pred[i] = -1
        if y_pred[i] == testlabel[i]:
            correct_count += 1
    return y_pred, correct_count / n1


n = X.shape[0]
foldsize = n // 5
with open("cross_validation.txt", "w") as f:
    for lamda in [1e-4, 0.01, 0.1, 0.5, 1, 5, 10, 100, 1000, 5000, 10000]:
        f.write("when lambda is {}, ".format(lamda))
        avg_accuracy = 0.0
        for i in xrange(5):
            begin = i * foldsize
            end = begin + foldsize
            traindata = np.row_stack((X[:begin, :], X[end:, :]))
            trainlabel = np.concatenate((y[:begin], y[end:]))
            testdata = X[begin:end, :]
            testlabel = y[begin:end]
            (y_pred, accuracy) = lsClassifier(traindata, trainlabel, testdata, testlabel, lamda)
            avg_accuracy += accuracy
        avg_accuracy /= 5
        f.write("the average accuracy is {}.\n".format(avg_accuracy))

'''
traindata = X[0:int(0.8 * n), :]
trainlabel = y[0:int(0.8 * n)]
testdata = X[int(0.8 * n):, :]
testlabel = y[int(0.8 * n):]
lsClassifier(traindata, trainlabel, testdata, testlabel, 0.1)
'''