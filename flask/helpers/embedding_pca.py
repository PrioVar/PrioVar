import torch
from os import path
from sklearn.decomposition import PCA
import numpy as np

# read the gene_dict.pt, hpo_dict.pt, and disease_dict.pt files
# into gene_dict, hpo_dict, and disease_dict dictionaries
gene_dict = torch.load(path.join('../data', 'gene_dict.pt'))
hpo_dict = torch.load(path.join('../data', 'hpo_dict.pt'))
disease_dict = torch.load(path.join('../data', 'disease_dict.pt'))

# read the embeddings from data/embedding_results/weighted_node2vec_p0.5_q2_n5_v256.txt
embeddings = np.loadtxt(
    "../data/embedding_results/node_embeddings_p0.5_q2_n5_v256.txt", skiprows=1
)

# NOTE: the first column indicates the index of the node,
# ranging from 0 to the number of nodes - 1

# get the embeddings where the first column is between 0 and len(gene_dict) - 1
gene_embeddings = embeddings[(embeddings[:, 0] >= 0) & (embeddings[:, 0] < len(gene_dict))]
gene_embeddings = gene_embeddings[:, 1:]  # remove the first column

dimension_list = [2, 4, 8, 16, 32, 64, 128, 256]

for dimension in dimension_list:
    pca = PCA(n_components=dimension)
    gene_embeddings_pca = pca.fit_transform(gene_embeddings)
    print(f"Explained variance ratio for dimension {dimension}: {sum(pca.explained_variance_ratio_):.2f}")
    print(f"Gene embeddings PCA shape for dimension {dimension}: {gene_embeddings_pca.shape}")


def apply_pca(embeddings: np.ndarray, dimension: int) -> np.ndarray:
    pca = PCA(n_components=dimension)
    return pca.fit_transform(embeddings)


# apply PCA with n_components=2 and n_components=4 to gene_embeddings and save
gene_embeddings_pca = apply_pca(gene_embeddings, 2)
np.savetxt("../data/embedding_results/gene_embeddings_pca2.txt", gene_embeddings_pca)

gene_embeddings_pca = apply_pca(gene_embeddings, 4)
np.savetxt("../data/embedding_results/gene_embeddings_pca4.txt", gene_embeddings_pca)

