from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import string
import os
import numpy as np

X = []
y = []
dim = 100
wv = KeyedVectors.load_word2vec_format("../glove.6B.100d.txt", binary=False)

def getFeature(comment):
    v = np.zeros(dim)
    count = 0
    for word in comment.split():
        if word in wv:
            v += wv[word]
            count += 1
    if count > 0:
        v /= count
    return v

'''
l = os.listdir("../original")
l.sort(key=lambda x:int(x[:-4]))
trantab = string.maketrans(string.punctuation, ' ' * len(string.punctuation))
with open("input.txt", "w") as g:
    for fname in l:
        with open(os.path.join("../original", fname)) as f:
            for line in f:
                line = line.translate(trantab)
                g.write(line + '\n')
'''

with open("input.txt") as f:
    for line in f:
        x = getFeature(line)
        X.append(x)

X = np.array(X)
with open("data") as f:
    for line in f:
        score = int((line.split())[0])
        if score >= 7:
            y.append(1)
        else:
            y.append(-1)
y = np.array(y)
