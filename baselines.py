from main import extract_word_features
from collections import defaultdict, Counter
from etcdata import speech_verbs
from dep_trees import generate_tree
from itertools import chain
from matplotlib import pyplot
import pickle
import networkx as nx
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

def random_baseline():
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)

    lbs_gold = []
    random = []
    c = 0
    for p in data:
        _, lp = extract_word_features(p)
        c += len(lp)
        lp = [int(x) for x in lp]
        lbs_gold += lp

    lbs_gold = np.array(lbs_gold)
    print(Counter(lbs_gold))

    lbs_random = [1 if np.random.random() < 0.156620222 else 0 for x in range(len(lbs_gold))]
    print(Counter(lbs_random))
    
    
    f1 = f1_score(lbs_gold, lbs_random)
    pr = precision_score(lbs_gold, lbs_random)
    re = recall_score(lbs_gold, lbs_random)

    print('f1', f1)
    print('pr', pr)
    print('re', re)

def h2_baseline():
    with open('.data/data.pickle', 'rb') as f:
        data = pickle.load(f)

    lbs_gold = []
    #random = []
    c = 0
    baseline_l = []
    for p in data:
        _, lp = extract_word_features(p)
        lbs_gold += lp
        # ignore punct
        capture = False
        for j, (label, word, word_info) in enumerate(p):
            
            if word_info[2].lower() in speech_verbs:
                baseline_l.append(1)
                capture = True
            else:
                if capture:
                    #if word_info[2] in ['.','?','!']:
                    if word_info[3] == 'PUNCT':
                        capture = False
                        continue
                    else:
                        if word_info[3] != 'PUNCT':
                            baseline_l.append(1)
                else:
                    if word_info[3] != 'PUNCT':
                        baseline_l.append(0)
                    else:
                        continue
                    
    lbs_gold = np.array(lbs_gold)
    baseline_l = np.array(baseline_l)
    print(len(lbs_gold), len(baseline_l))
    f1 = f1_score(lbs_gold, baseline_l)
    pr = precision_score(lbs_gold, baseline_l)
    re = recall_score(lbs_gold, baseline_l)

    
    print('pr', pr)
    print('re', re)
    print('f1', f1)
        
def h1_baseline():
    with open('./data/data.pickle', 'rb') as f:
        data = pickle.load(f)

    lbs_gold = []
    #random = []
    c = 0
    baseline_l = []
    for p in data:
        _, lp = extract_word_features(p)
        lbs_gold += lp
        # ignore punct
        capture = False
        for j, (label, word, word_info) in enumerate(p):
            
            if word_info[2].lower() in speech_verbs:
                baseline_l.append(1)
                capture = True
            else:
                if capture:
                    if word_info[2] in ['.','?','!']:
                        #capture = False
                        continue
                    else:
                        if word_info[3] != 'PUNCT':
                            baseline_l.append(1)
                else:
                    if word_info[3] != 'PUNCT':
                        baseline_l.append(0)
                    else:
                        continue
                    
    lbs_gold = np.array(lbs_gold)
    baseline_l = np.array(baseline_l)
    print(len(lbs_gold), len(baseline_l))
    f1 = f1_score(lbs_gold, baseline_l)
    pr = precision_score(lbs_gold, baseline_l)
    re = recall_score(lbs_gold, baseline_l)


    print('pr', pr)
    print('re', re)
    print('f1', f1)

if __name__ == '__main__':
    h2_baseline()
