## simple graph viz
import scipy.sparse
import operator
from collections import defaultdict
import numpy as np
import glob
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from py3plex.algorithms.community_detection import community_wrapper as cw
from py3plex.core import multinet
from py3plex.visualization.multilayer import *
from py3plex.visualization.colors import colors_default
from collections import Counter
from py3plex.wrappers import train_node2vec_embedding
from py3plex.visualization.embedding_visualization import embedding_visualization,embedding_tools

if __name__ == "__main__":

    graph_name = "../graphs/graph_en_sl.gpickle"
    graph = nx.read_gpickle(graph_name)
    comNet = multinet.multi_layer_network().load_network(graph,directed=True,input_type='nx')

    comNet.save_network("./test.edgelist")

    ## call a specific n2v compiled binary
    train_node2vec_embedding.call_node2vec_binary("./test.edgelist","./test_embedding.emb",binary="./node2vec",weighted=False)

    ## preprocess and check embedding
    comNet.load_embedding("./test_embedding.emb")
    output_positions = embedding_tools.get_2d_coordinates_tsne(comNet,output_format="pos_dict")

    ## custom layouts are part of the custom coordinate option
    layout_parameters = {}
    layout_parameters['pos'] = output_positions ## assign parameters
    network_colors, graph = comNet.get_layers(style="hairball")
    hairball_plot(graph,network_colors,layout_algorithm="custom_coordinates",layout_parameters=layout_parameters)
    plt.show()

