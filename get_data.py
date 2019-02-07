from tokenizer import build_sentences as tok
from itertools import chain

def read_stagger(spath):
    with open(spath) as f:
        return [x.split() for x in f.read().split('\n') if x]

def get_gold_labels(line):
    line = line.rstrip()
    line_string = ' '.join([x for x in line.split(' ') if x != '<NC>'])
    line = [[y for y in x if y not in ['<', 'NC']] for x in tok(line)]

    isnc = False
    wordl = []
    label = []
    for sentence in line:
        sentl = []
        sentw = []
        for word in sentence:
            if word == '>':
                if isnc == True:
                    isnc = False
                else:
                    isnc = True
                continue
            else:
                sentl.append(isnc)
                sentw.append(word)
        wordl.append(sentw)
        label.append(sentl)
    return wordl, label        

def read_file(data_wgold):
    gold_standard = []
    with open(data_wgold) as f:
        for ln in f:
            if not ln.split():
                continue
            data, label = get_gold_labels(ln)
            gold_standard.append((data, label))
    return gold_standard

def stagger_mapping(gold_data, stagger_data):
    data_mapping = []
    wi = 0
    for wsent, labels in gold_data:
        nline = []
        for sentencew, sentencel in zip(wsent, labels):
            nsent = []
            for word, label in zip(sentencew, sentencel):
                nline.append((label, word, stagger_data[wi]))
                if stagger_data[wi][1] != word:
                    print(stagger_data[wi], word)
                    print(sentencew)
                assert stagger_data[wi][1] == word
                wi += 1
        data_mapping.append(nline)


    return data_mapping

def get_data():
    gold_data = read_file('/home/adam/git/dialogue_structure_v2/all_data/data_wgold.txt')
    stagger_data = read_stagger('/home/adam/git/dialogue_structure_v2/all_data/tagged_data.conll')
    return stagger_mapping(gold_data, stagger_data)


if __name__ == '__main__':
    gold_data = read_file('/home/adam/git/dialogue_structure_v2/all_data/data_wgold.txt')
    stagger_data = read_stagger('/home/adam/git/dialogue_structure_v2/all_data/tagged_data.conll')
    data_set = stagger_mapping(gold_data, stagger_data)


