### simple keyword detector!

#from construct_graph import *
import nltk
import string
from nltk.corpus import stopwords as stpw
from nltk import word_tokenize
from nltk.corpus import stopwords
import operator
from nltk.stem import WordNetLemmatizer
import networkx as nx
import numpy as np

import re
import pandas
from nltk.corpus import wordnet as wn
from tqdm import tqdm

# def get_taxonomy():
#     G = nx.DiGraph()
#     try:
#         wn.all_synsets
#     except LookupError as e:
#         import nltk
#         nltk.download('wordnet')

#     # make sure each edge is included only once
#     edges = set()
#     for synset in tqdm(wn.all_synsets(pos='n')):
#         # write the transitive closure of all hypernyms of a synset to file
#         for hyper in synset.closure(lambda s: s.hypernyms()):
#             edges.add((synset.name(), hyper.name()))

#         # also write transitive closure for all instances of a synset
#         for instance in synset.instance_hyponyms():
#             for hyper in instance.closure(lambda s: s.instance_hypernyms()):
#                 G.edges.add(instance.name(), hyper.name())
#                 for h in hyper.closure(lambda s: s.hypernyms()):
#                      G.edges.add(instance.name(), h.name())
#     return G

def corpus_graph(language_file,limit_range=300000,verbose=False,lemmatizer=None,stopwords=None, min_char = 3,stemmer=None,token_len=1):

    G = nx.DiGraph()
    ctx = 0
    reps = False
    dictionary_with_counts_of_pairs = {}
    with open(language_file) as lf:
        for line in lf:
            stop = list(string.punctuation)
            line = line.strip()
            line = [i for i in word_tokenize(line.lower()) if i not in stop]
            
            if not stopwords is None:
                line = [w for w in line if not w in stopwords]
                
            if not stemmer is None:
                line = [stemmer.stem(w) for w in line]
                
            if not lemmatizer is None:
                line = [lemmatizer.lemmatize(x) for x in line]
                
            line = [x for x in line if len(x) > min_char]
            if token_len == 1:
                if len(line) > 1:
                    ctx+=1
                    if ctx % 15000 == 0:
                        logging.info("Processed {} sentences.".format(ctx))
                    if ctx % limit_range == 0:
                        break
                    for enx, el in enumerate(line):
                        edge_directed = None
                        if enx > 0:                            
                            edge_directed = (line[enx-1],el)
                            if edge_directed[0] != edge_directed[1]:
                                G.add_edge(edge_directed[0], edge_directed[1])
                            else:
                                edge_directed = None
                        if edge_directed:
                            if edge_directed in dictionary_with_counts_of_pairs:
                                dictionary_with_counts_of_pairs[edge_directed] += 1
                                reps = True
                            else:
                                dictionary_with_counts_of_pairs[edge_directed] = 1

    ## assign edge properties.
    for edge in G.edges(data=True):
        try:
            edge[2]['weight'] = dictionary_with_counts_of_pairs[(edge[0],edge[1])]
        except Exception as es:
            raise (es)
    if verbose:
        print(nx.info(G))
    return (G,reps)

def find_keywords(document, limit_num_keywords = 10, lemmatizer=None, num_sentences=30000,order=1,double_weight_threshold=2):
    
    lemmatizer = WordNetLemmatizer()
    stopwords = set(stpw.words('english'))
    klens = {}
    weighted_graph,reps = corpus_graph(document,limit_range=num_sentences,lemmatizer=lemmatizer,stopwords=stopwords,token_len=1)
    pgx = nx.pagerank(weighted_graph)
    keywords_with_scores = sorted(pgx.items(), key=operator.itemgetter(1),reverse=True)
    kw_map = dict(keywords_with_scores)
    if reps:
        frequent_pairs = []
        for edge in weighted_graph.edges(data=True):
            if edge[0] != edge[1]:
                if "weight" in  edge[2]:
                    if edge[2]['weight'] > double_weight_threshold:
                        frequent_pairs.append(edge[0:2])
        higher_order_list = []
        for pair in frequent_pairs:        
            w1 = pair[0]
            w2 = pair[1]        
            if w1 in kw_map and w2 in kw_map:                               
                score = np.max([kw_map[w1],kw_map[w2]])
                higher_order_list.append((w1+" "+w2,score))
    else:
        higher_order_list = []                
    total_kws = sorted(keywords_with_scores+higher_order_list,key=operator.itemgetter(1),reverse=True)[0:limit_num_keywords]
    return total_kws

if __name__ == "__main__":

    ## fajl z dokumentom.
    import glob
    for text_file_name in glob.glob("../keyword_baselines/docsutf8/*"):
        # text_file_name = "../keyword_baselines/docsutf8/crime-20943227.txt"
        keywords_single_word = find_keywords(text_file_name,limit_num_keywords=10,order=1)
        
    # keywords_double_word = find_keywords(text_file_name,limit_num_keywords=10,order=2)
    # print(keywords_double_word)
