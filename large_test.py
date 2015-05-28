import glob
import os
import numpy as np

# sklearn
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

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

# baseline:
knn_classifier = KNeighborsClassifier(n_neighbors=1, algorithm="brute", metric="minkowski", p=1)

# network:
def get_nnet_classifier(input_dim, hidden_dim, output_dim):
    # we use constructor fundtion here, to make sure we get a new model each time
    nnet_classifier = Sequential()
    nnet_classifier.add(Dense(input_dim=input_dim, output_dim=hidden_dim, init="uniform"))
    nnet_classifier.add(Activation("relu"))
    nnet_classifier.add(Dense(input_dim=hidden_dim, output_dim=output_dim, init="uniform"))
    nnet_classifier.add(Activation("softmax"))
    nnet_classifier.compile(loss='categorical_crossentropy', optimizer='adadelta')    
    return nnet_classifier

true_y, nnet_pred_y, neighb_pred_y = [], [], []
for target_title in sorted(list(set(titles))):#[:5]:
    print("Leaving out "+target_title)
    train_X, train_y, test_X, test_y = [], [], [], [] 
    for doc_author, doc_vec, doc_title in zip(author_y, X, titles):
        if doc_title == target_title:
            test_X.append(doc_vec); test_y.append(doc_author)
        else:
            train_X.append(doc_vec); train_y.append(doc_author)
    # let's keep track of the correct labels:
    true_y.extend(test_y)
    train_X = np.asarray(train_X); test_X = np.asarray(test_X)
    # colllect baseline results:
    knn_classifier.fit(train_X, train_y)
    neighb_pred_y.extend(knn_classifier.predict(test_X))
    # now deep net:
    train_y = np_utils.to_categorical(train_y, nb_classes=len(enc.classes_))
    test_y = np_utils.to_categorical(test_y, nb_classes=len(enc.classes_))
    nnet_classifier = get_nnet_classifier(input_dim=train_X.shape[1], hidden_dim=100, output_dim=len(enc.classes_))
    nnet_classifier.fit(train_X, train_y, show_accuracy=True, nb_epoch=8, batch_size=1)
    nnet_pred_y.extend(nnet_classifier.predict_classes(test_X, batch_size=1))
    print("::::::::::::::::::")

print("And the winner is...")
print("Baseline\n\t-Accuracy: "+str(accuracy_score(true_y, neighb_pred_y)))
print("\t-F1-score: "+str(f1_score(true_y, neighb_pred_y, pos_label=None, average="weighted")))
print("Network\n\t-Accuracy: "+str(accuracy_score(true_y, nnet_pred_y)))
print("\t-F1-score: "+str(f1_score(true_y, nnet_pred_y, pos_label=None, average="weighted")))
