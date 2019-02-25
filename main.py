from get_data import get_data
from collections import defaultdict, Counter
from etcdata import speech_verbs
from dep_trees import generate_tree
from itertools import chain
from matplotlib import pyplot
import pickle
import networkx as nx
import numpy as np
import numpy.random as npr
from random import seed, random

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, r2_score

def extract_word_features(dialogue_line, window_len):
    featureset, labelset = [], []

    sentence_indices = []
    sentence_stagger = []
    units = [[]]

    # structure each dialogue line into word units and sentences 
    for j, (l, w, word_info) in enumerate(dialogue_line):
        if word_info[0] == '1':
            sentence_indices.append([j])
            sentence_stagger.append([word_info])
        else:
            sentence_indices[-1].append(j)
            sentence_stagger[-1].append(word_info)

        if j == 0:
            units[-1].append(j)
            continue
        if word_info[3] == 'PUNCT':
            units[-1].append(j)
            units.append([])
        else:
            units[-1].append(j)
    
    units = [x for x in units if x]
    
    for i, (label, word, info) in enumerate(dialogue_line):
        if info[3] == 'PUNCT':
            # dont predict PUNCT
            continue
        else:

            for k, u in enumerate(units):
                if i in u:
                    word_unit = k
                    
            for k, s in enumerate(sentence_indices):
                if i in s:
                    word_sentence = k
                    word_sentence_ending = dialogue_line[s[-1]][2][1]
            
            sentence_tree = generate_tree(sentence_stagger[word_sentence])
            
            f = word_features(info, 'current=', str(i)+'=', sentence_tree)
            f_num = len(f)

            # iterate over context
            for j in range(1,window_len):
                # next word
                try:
                    f += word_features(dialogue_line[i+j][2])
                except:
                    f += [0]*f_num

                # previous word
                try:
                    f += word_features(dialogue_line[i-j][2])
                except:
                    f += [0]*f_num

            f += [word_sentence_ending] 
            f += [word_sentence]
            f += [word_unit]

            featureset.append({f'feature_{i}':x for (i, x) in enumerate(f)})
            labelset.append(label)

    return featureset, labelset


def word_features(line):
    fs = []
    fs.append(line[2].lower() in speech_verbs) # is speech verb
    fs.append(line[2].lower()) # word form
    fs.append(line[1] in ['?','!']) # is exclamation
    fs.append(line[3]) # pos
    fs.append(line[3] == 'PUNCT') # pos is punct
    fs.append(line[4]) # grammar 
    fs.append(line[-3]) # dep relation
    fs.append(line[-3] == 'root') # is root

    return fs

def train_test_tokencv(data, labels):
    # remove dev
    cutoff = int(len(labels)*0.05)
    data = data[cutoff:]
    labels = labels[cutoff:]

    data = DictVectorizer().fit_transform(list(chain.from_iterable(data)))
    labels = list(chain.from_iterable(labels))

    print('lbs', len(labels))
    
    folds = 10
    model = LogisticRegression(max_iter=10, solver='liblinear')
    score = cross_val_score(model, X=data, y=labels, cv=folds, scoring=s)
    print(s, sum(score)/folds)

def cross_val_split(data, labels, mseed, cv=10, ratio=0.1):
    
    seed(mseed)
    
    indices = set([i for (i,x) in enumerate(labels)])
    # sentences with/without narration
    t_sentences = np.array([i for (i, x) in enumerate(labels) if np.any(x)])
    f_sentences = np.array([i for i in indices if i not in t_sentences])

    # fold seeds
    seeds = npr.randint(100, size=cv)

    for i in range(cv):
        seed(seeds[i])
        
        te_s = int((len(data)*ratio)/2)
        test = set(npr.choice(t_sentences, te_s, replace=False)).union(
            set(npr.choice(f_sentences, te_s, replace=False)))
        train = indices.difference(test)

        train_x = list(map(lambda x: data[x], train))
        train_y = list(map(lambda x: labels[x], train))

        test_x = list(map(lambda x: data[x], test))
        test_y = list(map(lambda x: labels[x], test))

        yield test_x, test_y, train_x, train_y
    
def train_test_sentencecv(data, labels, mseed):
    # fit vectorizer
    vectorizer = DictVectorizer().fit(list(chain.from_iterable(data)))

    # remove dev
    cutoff = int(len(labels)*0.05)
    data = data[cutoff:]
    labels = labels[cutoff:]
    
    folds = 10
    cvs = cross_val_split(data, labels, mseed, folds)

    analysis = np.zeros((folds, 4))
    score = np.zeros((folds,3))
    for i, (test_x, test_y, train_x, train_y) in enumerate(cvs):
        sents = [x for x in test_y]

        test_x = vectorizer.transform(list(chain.from_iterable(test_x)))
        test_y = list(chain.from_iterable(test_y))

        train_x = vectorizer.transform(list(chain.from_iterable(train_x)))
        train_y = list(chain.from_iterable(train_y))

        model = LogisticRegression(max_iter=500,
                                   solver='liblinear',
                                   random_state=mseed)
        
        model.fit(train_x, train_y)
        r = model.predict(test_x)

        pred_sents = []
        preds = iter(list(r))
        for x in sents:
            pred_sents.append([next(preds) for _ in range(len(x))])
        
        score[i] = np.array([precision_score(test_y, r),
                             recall_score(test_y, r),
                             f1_score(test_y, r)])
        
    return np.array([np.mean(score[:,0]),
                     np.mean(score[:,1]),
                     np.mean(score[:,2])])
    
def dev_train_test(data, labels):
    vectorizer = DictVectorizer().fit(list(chain.from_iterable(data)))
    
    cutoff = int(len(data)*0.05)
    trainX, trainY = data[cutoff:], labels[cutoff:]
    testX, testY = data[:cutoff], labels[:cutoff]

    trainX = vectorizer.transform(list(chain.from_iterable(trainX)))
    trainY = list(chain.from_iterable(trainY))
    testX = vectorizer.transform(list(chain.from_iterable(testX)))
    testY = list(chain.from_iterable(testY))

    model = LogisticRegression(max_iter=100, solver='liblinear')
    model.fit(trainX, trainY)

    predY = model.predict(testX)

    score = f1_score(testY, predY)
    print(score)

def window_len_test():
    with open('./data/data.pickle', 'rb') as f:
        data = pickle.load(f)
    print('sentences in data:', len(data))

    window_len = 7
    score = np.zeros((len(range(0,10)),3))

    mseed = 6
    for window_len in range(0,10):
        featureset = []
        labelset = []
        for p in data:
            fp, lp = extract_word_features(p, window_len)
            featureset.append(fp)
            labelset.append(lp)
    
        score[window_len] = train_test_sentencecv(featureset, labelset, mseed)        

    for wlen, score in enumerate(score):
        print(wlen, score)

def test():
    with open('./data/data.pickle', 'rb') as f:
        data = pickle.load(f)
    print('sentences in data:', len(data))

    window_len = 4
    score = np.zeros((len(range(0,10)),3))

    mseed = 6
    featureset = []
    labelset = []
    for p in data:
        fp, lp = extract_word_features(p, window_len)
        featureset.append(fp)
        labelset.append(lp)
    
    score = train_test_sentencecv(featureset, labelset, mseed)
    print(score)

def test_dev():
    with open('./data/data.pickle', 'rb') as f:
        data = pickle.load(f)
    print('sentences in data:', len(data))

    featureset = []
    labelset = []
    for p in data:
        fp, lp = extract_word_features(p, 4)
        featureset.append(fp)
        labelset.append(lp)

    dev_train_test(featureset, labelset)
        
if __name__ == '__main__':
    #test_dev()
    run_tests()
    



 
