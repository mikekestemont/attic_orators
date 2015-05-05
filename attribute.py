import glob
import codecs
import os
import numpy as np

SAMPLE_SIZE = 1000

authors, titles, texts = [], [], []
for filename in glob.glob("corpus/*.txt"):
    text = codecs.open(filename, 'r', 'utf-8').read()
    words = text.split()
    # sample:
    start_idx, end_idx, cnt = 0, SAMPLE_SIZE, 1
    author, title = os.path.splitext(
                       os.path.basename(filename))[0].split('_')
    while end_idx <= len(words):
        authors.append(author)
        titles.append(title)
        texts.append(words[start_idx:end_idx])
        cnt+=1
        start_idx+=SAMPLE_SIZE
        end_idx+=SAMPLE_SIZE

def identity(words):
    return words

.toarray()
print X.shape

# weigh X for Burrows's Delta:
stds = np.std(X, axis=0)
print stds.shape
X *= stds
print X.shape

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(authors)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=1)









