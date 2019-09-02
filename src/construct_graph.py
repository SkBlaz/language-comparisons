## parse files and construct a graph
## http://opus.nlpl.eu/DGT.php

import glob
import operator
from collections import Counter
import pandas as pd
import itertools
import networkx as nx
import os
import tqdm
from nltk import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import string
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)
from py3plex.algorithms.community_detection import community_wrapper as cw
from py3plex.core import multinet

def core_network_statistics(G,labels=None,name="example"):

    nodes = len(G.nodes())
    edges = len(G.edges())
    logging.info("Graph with {} nodes and {} edges.".format(nodes,edges))
    ccc = len(list(nx.connected_components(G.to_undirected())))

    try:
        cc = nx.average_clustering(G.to_undirected())
    except Exception as es:
        logging.info(es)
        cc = None

    try:
        dx = nx.density(G)
    except:
        dx = None

    comNet = multinet.multi_layer_network().load_network(G,directed=True,input_type='nx')
    try:
        partition = cw.infomap_communities(comNet, binary="./Infomap", multiplex=False, verbose=False,iterations = 100)
        partition = [v for k,v in partition.items()]
        num_communities = len(set(partition))
        max_community_size = Counter(partition)
    except:
        max_communities = None
        num_communities = None
    degs = [x[1] for x in G.degree([n for n in G.nodes()])]
    mean_degree = np.mean(degs)
    counts = max(nx.connected_components(G.to_undirected()), key=len)
    counter_dict = dict(max_community_size)    
    mean_com= np.mean([y for x,y in counter_dict.items()])
    max_com = np.max([y for x,y in counter_dict.items()])
    logging.info("max com size {}, mean com size {}".format(max_com,mean_com))
    
   # logging.info("Computing diameter..")
   # diameter = nx.diameter(G.subgraph(max_com))
    
    point = {"Name": name.replace("DGT.",""),
             "nodes":nodes,
             "edges":edges,
             "max_com" : max_com,
             "mean_com": mean_com,
             "degree":mean_degree,
             "connected components":ccc,
             "clustering coefficient":cc,
             "density":dx}
    return point

def corpus_graph(language_file,limit_range=250000,verbose=True,lemmatizer=None,stopwords=None, min_char = 0,stemmer=None):

    G = nx.DiGraph()
    ctx = 0
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
            if len(line) > 3:
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
            pass

    if verbose:
        print(nx.info(G))
    return G
            
            
def read_sentence_files(pair,datafolder):

    """
    A method which read the graphs either from disks or constructs them.
    """
    
    lans = pair.split(".")[-1].split("-")
    first_language = datafolder+pair+"."+lans[0]
    second_language = datafolder+pair+"."+lans[1]
    try:
        graph_first = nx.read_gpickle("../graphs/graph_{}.gpickle".format(lans[0]+"|"+lans[1]))
    except Exception as es:
        logging.info(es)
        graph_first = corpus_graph(first_language)
        nx.write_gpickle(graph_first,"../graphs/graph_{}.gpickle".format(lans[0]+"|"+lans[1]))
    try:
        graph_second = nx.read_gpickle("../graphs/graph_{}.gpickle".format(lans[1]+"|"+lans[0]))
    except Exception as es:
        logging.info(es)
        graph_second = corpus_graph(second_language)
        nx.write_gpickle(graph_second,"../graphs/graph_{}.gpickle".format(lans[1]+"|"+lans[0]))
    return (graph_first,graph_second,lans[0],lans[1])

def visualize_results(df):

    df[['First language','Second language']] = df.name.str.split("-",expand=True,)
    candidate_diff_attr = ["max_com","mean_com","density","connected components","clustering coefficient","degree"]
    for cand in candidate_diff_attr:
        dfx = df.pivot("First language","Second language",cand)
        fig = plt.figure()
        fig = sns.heatmap(dfx)
        plt.savefig("../result_images/{}".format(cand))
        plt.clf()

def validate_language_pairs(pairs):

    language_pairs = [x.split(".")[1].split("-") for x in pairs]
    unique_languages = set([x[0] for x in language_pairs] + [x[1] for x in language_pairs])
    all_possible = list(itertools.combinations(unique_languages,2))
    logging.info("all pairs {}, possible pairs {}".format(len(language_pairs),len(all_possible)))
    missing = []
    for el in all_possible:
        not_present = True
        for lp in language_pairs:
            lps = set(lp)
            els = set(el)
            if lps == els:
                not_present = False
        if not_present:
            logging.info("Missing language {}".format(el))
            
if __name__ == "__main__":

    datafolder = "../data/*"
    all_language_pairs = set()
    plot_only = False
    if plot_only == False:
        for fx in glob.glob(datafolder):
            if "zip" not in fx:
                language_pair = ".".join(fx.split("/")[-1].split(".")[0:2])
                if "DGT" in language_pair:
                    all_language_pairs.add(language_pair)
        validate_language_pairs(all_language_pairs)
        all_stats = []
        all_ind =  pd.DataFrame()
        all_language_pairs = list(all_language_pairs)
        for language_pair in all_language_pairs:
            try:
                logging.info("Parsing {}".format(language_pair))
                graph_first,graph_second, l1,l2 = read_sentence_files(language_pair,"../data/")
                l1 = l1.replace("DGT.","")
                stats_frame_first = core_network_statistics(graph_first,name=l1)
                l2 = l2.replace("DGT.","")
                stats_frame_second = core_network_statistics(graph_second,name=l2)
                merged = {"name":language_pair.replace("DGT.","")}
                for k,_ in stats_frame_first.items():
                    if k != "Name":                
                        merged[k] = stats_frame_second[k] - stats_frame_first[k]
                        all_ind.append(stats_frame_first,ignore_index=True)
                        all_ind.append(stats_frame_second,ignore_index=True)
                all_stats.append(merged)                
            except Exception as es:
                print(es)                
        dx = pd.DataFrame(all_stats)
        all_ind.to_csv("../results/individual.tsv",sep="\t")
        dx.to_csv("../results/comparison_dataframe.tsv",sep="\t")
