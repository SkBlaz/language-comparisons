import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse

w2w_matrix = np.load("w2w_matrix.npy",allow_pickle=True)
t2g_model = np.load("t2g_matrix.npy",allow_pickle=True)
adjacency  = sparse.load_npz("adj_matrix.npz")
ft_matrix  = np.load("ft_matrix.npy",allow_pickle=True)

intra_distance_w2w = np.load("dists_w2w.npy")
intra_distance_t2n = np.load("dists_t2n.npy")
intra_distance_ft = np.load("dists_ft.npy")

intra_dists_w2w = np.sort(intra_distance_w2w)
intra_dists_t2n = np.sort(intra_distance_t2n)
intra_dists_ft = np.sort(intra_distance_ft)
print(np.mean(intra_dists_w2w),np.mean(intra_dists_t2n),np.mean(intra_dists_ft))
sns.distplot(intra_dists_ft, color="orange",label="FastText")
sns.distplot(intra_dists_w2w, color="green",label="word2vec")
sns.distplot(intra_dists_t2n, color="black",label="text2net2vec")
plt.xlabel("Variability within a sentence (context)")
plt.ylabel("Density")
plt.legend()    
plt.savefig("../result_images/COMPARISON_w2w.png")
plt.clf()

print("UMAP fitting part.")
pca_w2w = umap.UMAP().fit_transform(w2w_matrix)
pca_t2g = umap.UMAP().fit_transform(t2g_model)
pca_adj = umap.UMAP().fit_transform(adjacency)
pca_ft = umap.UMAP().fit_transform(ft_matrix)

plt.scatter(pca_t2g[:,0],pca_t2g[:,1],alpha=0.3,marker="o",s=5,label="text2net + NetMF",color="black")
plt.xlabel("UMAP dim 1")
plt.ylabel("UMAP dim 2")
#plt.axis('scaled')
plt.savefig("../result_images/EMBEDDING_t2g_{}".format("w1"))
plt.clf()

plt.scatter(pca_w2w[:,0], pca_w2w[:,1], alpha=0.3,marker="o",s=5,label="word2vec",color="black")
plt.xlabel("UMAP dim 1")
plt.ylabel("UMAP dim 2")
#plt.axis('scaled')
plt.savefig("../result_images/EMBEDDING_w2w_{}".format("w1"))
plt.clf()

plt.scatter(pca_adj[:,0], pca_adj[:,1],alpha=0.3,marker="o",s=5,label="Laplacian",color="black")
plt.xlabel("UMAP dim 1")
plt.ylabel("UMAP dim 2")
#plt.axis('scaled')
plt.savefig("../result_images/EMBEDDING_adj_{}".format("w1"))
plt.clf()

plt.scatter(pca_ft[:,0], pca_ft[:,1],alpha=0.3,marker="o",s=5,label="Laplacian",color="black")
plt.xlabel("UMAP dim 1")
plt.ylabel("UMAP dim 2")
#plt.axis('scaled')
plt.savefig("../result_images/EMBEDDING_ft_{}".format("w1"))
plt.clf()
