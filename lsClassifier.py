from __future__ import division
import numpy as np
from getFeature import X, y

def lsClassifier(traindata, trainlabel, testdata, testlabel, lamda):
    n = traindata.shape[0]
    m = traindata.shape[1]
    print n, m
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
    ypred = testdata.dot(w)
    count_correct = 0
    for i in xrange(n1):
        if ypred[i] > 0:
            ypred[i] = 1
        else:
            ypred[i] = -1
        if ypred[i] == testlabel[i]:
            count_correct += 1
    print count_correct / n1

n = X.shape[0]
traindata = X[0:int(0.8 * n), :]
trainlabel = y[0:int(0.8 * n)]
testdata = X[int(0.8 * n):, :]
testlabel = y[int(0.8 * n):]
lsClassifier(traindata, trainlabel, testdata, testlabel, 0.001)