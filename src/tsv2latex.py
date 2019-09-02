## convert tsv results to latex results
import pandas as pd
import numpy as np
import networkx as nx
import tqdm
import glob

pdx = pd.read_csv("../results/comparison_dataframe.tsv",sep="\t")
pdx = pdx[["name","nodes","edges","degree","density","max_com","mean_com","clustering coefficient","connected components"]].sort_values("name")
pdx = pdx.round(3)
relative_nodes = []
relative_edges = []

graph_by_id = [x.split("/")[-1] for x in glob.glob("../graphs/*")]
graphs = {}
for gx in graph_by_id:
    core_name = gx.split("_")[1][0:2]
    graphs[core_name] = gx

for idx, row in tqdm.tqdm(pdx.iterrows()):

    lang1 = row['name'].split("-")[0]
    lang2 = row['name'].split("-")[1]
    
    graph1 = nx.read_gpickle("../graphs/{}".format(graphs[lang1]))
    graph2 = nx.read_gpickle("../graphs/{}".format(graphs[lang2]))

    num_nodesr = np.round((100*len(graph2.nodes()))/len(graph1.nodes()),2)
    num_edgesr = np.round((100*len(graph2.edges()))/len(graph1.edges()),2)
    
    relative_nodes.append(num_nodesr)
    relative_edges.append(num_edgesr)
    
pdx['relative_node_diff'] = relative_nodes
pdx['relative_edge_diff'] = relative_edges
pdx.to_latex("../results/latex_table.tex",index=False)

# pdx[["nodes","edges","degree","density","max_com","mean_com","clustering coefficient","connected components"]] = pdx[["nodes","edges","degree","density","max_com","mean_com","clustering coefficient","connected components"]].abs()
# pdx[['First language','Second language']] = pdx.name.str.split("-",expand=True,)
# pdx.to_csv("../results/clean_abs.tsv",sep="\t")
