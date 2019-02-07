from gold_standard import get_data
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
from sklearn.metrics import f1_score, precision_score, recall_score

def extract_word_features(dialogue_line):
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
                    word_unit_ending = dialogue_line[u[-1]][2][1]
                    
            for k, s in enumerate(sentence_indices):
                if i in s:
                    word_sentence = k
                    word_sentence_ending = dialogue_line[s[-1]][2][1]
            
            sentence_tree = generate_tree(sentence_stagger[word_sentence])
            
            f = word_features(info, 'current=', str(i)+'=', sentence_tree)
            f_num = len(f)

            # iterate over context
            for j in range(1,7):
                # next word
                try:
                    for m, s in enumerate(sentence_indices):
                        if i+j in s:
                            context_sentence_tree = generate_tree(sentence_stagger[m])

                    f += word_features(dialogue_line[i+j][2], 
                                       'following', 
                                       str(j),
                                       context_sentence_tree)
                except:
                    f += [0]*f_num

                # previous word
                try:
                    for m, s in enumerate(sentence_indices):
                        if i-j in s:
                            context_sentence_tree = generate_tree(sentence_stagger[m])

                    f += word_features(dialogue_line[i-j][2], 
                                       'previous', 
                                       str(j),
                                       context_sentence_tree)
                except:
                    f += [0]*f_num

            f += [word_sentence_ending] 
            #f += [word_unit_ending] 
            f += [word_sentence]
            f += [word_unit]

            featureset.append({f'feature_{i}':x for (i, x) in enumerate(f)})
            labelset.append(label)

    return featureset, labelset


def word_features(line, position, pos_num, sentence_tree=False):
    fs = []
    fs.append(line[2].lower() in speech_verbs) # is speech verb
    fs.append(line[2].lower()) # word form
    fs.append(line[1] in ['?','!']) # is exclamation
    fs.append(line[3]) # pos
    fs.append(line[3] == 'PUNCT') # pos is punct
    fs.append(line[4]) # grammar 

    #fine-grained dependency features, WiP,WiP,WiP
    dep_features = False
    if dep_features:
        word_node = '_'.join([line[0], line[1], line[-3]])
        dep_edges = sentence_tree.edges()
        dep_nodes = sentence_tree.nodes()

        in_relations = len([x for x in dep_edges if x[1] == word_node])
        fs.append(in_relations)

        parataxis = [x for x in dep_nodes if 'parataxis' in x]
        if parataxis:
            fs.append(sentence_tree.has_edge(word_node, parataxis[0]))
        
        root_node = [x for x in dep_nodes if 'root' in x]
        if root_node:
            fs.append(sentence_tree.has_edge(word_node, root_node[0]))

        fs.append(line[-3] == 'root')
        fs.append(line[-3] == 'parataxis')

        fs.append(len(dep_nodes)/len(dep_edges))

    else:
        fs.append(line[-3])
        fs.append(line[-3] == 'root') 

    return fs

def train_test_cvscore(data, labels):
    # remove dev
    cutoff = int(len(labels)*0.05)
    data = data[cutoff:]
    labels = labels[cutoff:]

    data = DictVectorizer().fit_transform(list(chain.from_iterable(data)))
    labels = list(chain.from_iterable(labels))

    print('lbs', len(labels))
    
    #for s in ['precision','recall','f1']:
    s = 'f1'
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
    
def train_test_s(data, labels, mseed):
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

        model = LogisticRegression(max_iter=100,
                                   solver='liblinear')
        
        model.fit(train_x, train_y)
        r = model.predict(test_x)
        #print(r)
        pred_sents = []
        preds = iter(list(r))
        for x in sents:
            pred_sents.append([next(preds) for _ in range(len(x))])

        #for x, y in zip(sents, pred_sents):
        #    print(len(x), len(y))

        from eval_stuff import sentence_level_eval
        analysis[i] = sentence_level_eval(sents, pred_sents)
        
        score[i] = np.array([precision_score(test_y, r),
                             recall_score(test_y, r),
                             f1_score(test_y, r)])
        #return 1


    print('full', np.mean(analysis[:,0]))
    print('part', np.mean(analysis[:,1]))
    print('errs', np.mean(analysis[:,2]))
    print('none', np.mean(analysis[:,3]))
    
    print('pr', np.mean(score[:,0]))
    print('re', np.mean(score[:,1]))
    print('f1', np.mean(score[:,2]))
    print()
    
    return np.array([np.mean(score[:,0]), np.mean(score[:,1]), np.mean(score[:,2])])
    
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

def run_tests(disable=False):
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)
    print('sentences in data:', len(data))

    score = np.zeros((6,3))
    # different meta seeds (controls the sentences in the test set)
    for i, mseed in enumerate([3,6,9,12,15,18]):
        featureset = []
        labelset = []
        for p in data:
            fp, lp = extract_word_features(p)
            featureset.append(fp)
            labelset.append(lp)
    
        score[i] = train_test_s(featureset, labelset, mseed)

    #print('--- pr', np.mean(score[:,0]))
    #print('--- re', np.mean(score[:,1]))
    #print('--- f1', np.mean(score[:,2]))
        

    
if __name__ == '__main__':
    run_tests()


 
