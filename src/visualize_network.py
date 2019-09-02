from py3plex.algorithms.community_detection import community_wrapper as cw
from py3plex.core import multinet
from py3plex.visualization.multilayer import *
from py3plex.visualization.colors import colors_default
from collections import Counter
import networkx as nx

parsed_net = nx.read_gpickle("../graphs/graph_sl|en.gpickle")

network = multinet.multi_layer_network().load_network(input_file=parsed_net,
                                                      directed=False,
                                                      input_type="nx")
network.basic_stats()  # check core imports

partition = cw.louvain_communities(network)
# select top n communities by size
top_n = 5
partition_counts = dict(Counter(partition.values()))
top_n_communities = list(partition_counts.keys())[0:top_n]

# assign node colors
color_mappings = dict(zip(top_n_communities, [x for x in colors_default if x != "black"][0:top_n]))

network_colors = [color_mappings[partition[x]] if partition[x] in top_n_communities else "black" for x in network.get_nodes()]


# visualize the network's communities!
hairball_plot(network.core_network,
              color_list=network_colors,
              layout_parameters={"iterations": 100},
              scale_by_size=True,
              layout_algorithm="force",
              legend=False)
plt.show()
