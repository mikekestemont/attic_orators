import glob
import os
import numpy as np

import matplotlib.pyplot as plt

# sklearn
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

# keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils

def slice(data_dir, sample_size):
    authors, titles, texts = [], [], []
    for filename in glob.glob(data_dir+"/*.txt"):
        text = open(filename, 'r').read()
        words = text.lower().split()
        # sample:
        start_idx, end_idx, cnt = 0, sample_size, 1
        author, title = os.path.splitext(
                           os.path.basename(filename.lower()))[0].split('_')
        while end_idx <= len(words):
            authors.append(author)
            titles.append(title)
            texts.append(words[start_idx:end_idx])
            # we update our counters:
            cnt += 1
            start_idx += sample_size
            end_idx += sample_size
    return authors, titles, texts

def identity(document):
    return document

authors, titles, texts = slice(data_dir="corpus", sample_size=500)
vectorizer = TfidfVectorizer(analyzer=identity, use_idf=False, max_features=2000)
X = vectorizer.fit_transform(texts).toarray()
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
author_y = enc.fit_transform(authors)

hidden_dim = 100
autoencoder = Sequential()
autoencoder.add(Dense(input_dim=X.shape[1], output_dim=hidden_dim, init="uniform"))
autoencoder.add(Activation("relu"))
autoencoder.add(Dense(input_dim=hidden_dim, output_dim=X.shape[1], init="uniform"))
autoencoder.add(Activation("softmax"))
autoencoder.compile(loss='categorical_crossentropy', optimizer='adadelta')

autoencoder.fit(X, X, show_accuracy=False, batch_size=1, nb_epoch=10)

# get projected version of the data:
print X.shape
weights = autoencoder.layers[0].get_weights()[0]
print weights.shape
X_projected = np.dot(X, weights)
print X_projected.shape

tsne = TSNE()
X_tsne = tsne.fit_transform(X_projected)
print X_tsne.shape

fig, ax1 = plt.subplots()
x1, x2 = X_tsne[:,0], X_tsne[:,1]
ax1.scatter(x1, x2, 100, edgecolors='none', facecolors='none');
for x, y, l in zip(x1, x2, [a[:3] for a in authors]):
    ax1.text(x, y, l, ha='center', va="center", size=10, color="darkgrey")
plt.savefig("tsne.pdf")









