## results visualization method
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from py3plex.core import multinet
import pandas as pd

colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
paletx = "vlag"
sns.set_palette(paletx)
def visualize_results(df):

    candidate_diff_attr = ["max_com","mean_com","density","connected components","clustering coefficient","degree","nodes","edges"]
    for cand in candidate_diff_attr:
        dfx = df.pivot("First language","Second language",cand).abs()
        fig = plt.figure()
        fig = sns.heatmap(dfx,cmap=paletx)
        plt.savefig("../result_images/{}".format(cand))
        plt.clf()
        
        # dfx = dfx.fillna(0)
        # fig = plt.figure()
        # fig = sns.clustermap(df[[cand,"First language","Second language"]], row_cluster=False,cmap=paletx)
        # plt.savefig("../result_images/clustermap_{}".format(cand))
        # plt.clf()
    

def visualize_graphs(df,measure="mean_com"):
    
    sns.distplot(df[measure])
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Distribution of {}".format(measure))
    plt.savefig("../result_images/dist_{}".format(measure))
    plt.clf()

if __name__ == "__main__":

    dx = pd.read_csv("../results/comparison_dataframe.tsv",sep="\t")
    dx[['First language','Second language']] = dx.name.str.split("-",expand=True,)
    visualize_results(dx)
    dx.to_csv("../results/clear_table.tsv")
    # for measure in ["max_com","mean_com","density","connected components","clustering coefficient","degree"]:
    #     dx = dx.round(2)
    #     dx[['First language','Second language',measure]].to_csv('../tmp/measure_{}.csv'.format(measure))        
    #     visualize_graphs(dx,measure)
