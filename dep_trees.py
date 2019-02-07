import networkx as nx
from matplotlib import pyplot as plt

def read_stagger(fpath):
    with open(fpath) as f:
        return [x.split('\t') for x in f.read().split('\n')]

def get_sentences(sdata):
    sentences = [[]]
    for x in sdata:
        if x == ['']:
            sentences.append([])
        else:
            sentences[-1].append(x)
    return sentences
        
def generate_tree(sentence):
    G = nx.DiGraph()

    for w in sentence:
        node_name = '_'.join([w[0], w[1], w[-3]])
        G.add_node(node_name)


    for i, word in enumerate(sentence):
        target_node = 'None'
        #print(word)
        if word[-3] == 'root':
            continue
        else:
            for j, w in enumerate(sentence):
                if i == j:
                    continue
                else:
                    if word[-4] == w[0]:
                        target_node = '_'.join([w[0], w[1], w[-3]])
            tag = word[-3]
            source_node = '_'.join([word[0], word[1], word[-3]])

            G.add_edge(source_node, target_node, tag=tag)
            #G[source_node][target_node]['tag'] = tag
    
    return G


def main():
    fpath = '/home/adam/git/dialogue_structure_v2/all_data/tagged_data.conll'
    data = read_stagger(fpath)
    sentences = get_sentences(data)

    g = generate_tree(sentences[1])
    print(g.edges(data='tag'))
    print(g.nodes())
    print(g.edges())
    nx.write_gexf(g, 'test1.gexf')
    g.all_simple_paths('1_-_punct', '4_?_punct')
    return g

if __name__ == '__main__':
    main()
