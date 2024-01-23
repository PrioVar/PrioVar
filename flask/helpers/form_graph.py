import torch
from os import path
from typing import Tuple
from hpo import read_hpo_from_json, process_nodes, process_edges
from gene_mapping import get_gene_mapping_dict


def get_hpo_terms_edges() -> Tuple:
    graph_id, meta, nodes, edges, property_chain_axioms = read_hpo_from_json()
    items = process_nodes(nodes)

    # only create a list of id's from items
    hpo_list = [item['id'] for item in items]

    # sort the hpo_list
    hpo_list.sort()

    hpo_edges = process_edges(edges)

    return hpo_list, hpo_edges


# read the combined genemania network into a tensor
# in each row, there's Gene_A, Gene_B, and the edge weight
with open(path.join('../data', 'COMBINED.DEFAULT_NETWORKS.BP_COMBINING.txt'), 'r') as f:
    combined_network = f.readlines()

# remove the header
combined_network = combined_network[1:]

# TODO: REMOVE THIS
# combined_network = combined_network[:10]

gene_set = set()
for row in combined_network:
    ls = row.strip().split('\t')
    gene_set.add(ls[0])
    gene_set.add(ls[1])

# get the dictionary using gene names
gene_mapping_dict = get_gene_mapping_dict()

# create the gene list and map it using the gene_mapping_dict and then sort it
gene_list = list(gene_set)
gene_list = [gene_mapping_dict[gene] for gene in gene_list]
gene_list.sort()

# create a dictionary to map gene names to indices
gene_dict = {}
for i in range(len(gene_list)):
    gene_dict[gene_list[i]] = i

hpo_list, hpo_edges = get_hpo_terms_edges()

hpo_dict = {}
for i in range(len(hpo_list)):
    hpo_dict[hpo_list[i]] = i + len(gene_list)

'''
 Fill in other nodes
'''

# TODO: CHANGE THIS LINE IF YOU ADD MORE NODES
num_edges = len(combined_network) + len(hpo_edges)

# create an edge_index tensor with size (2, num_edges)
edge_index = torch.zeros(2, num_edges)
edge_weight = torch.zeros(num_edges)

# fill in the network tensor
for i, row in enumerate(combined_network):
    ls = row.strip().split('\t')
    gene_a = gene_dict[ls[0]]
    gene_b = gene_dict[ls[1]]
    weight = float(ls[2])

    edge_index[0, i] = gene_a
    edge_index[1, i] = gene_b

    edge_weight[i] = weight

# fill in the tensor using hpo_edges
for i, edge in enumerate(hpo_edges):
    hpo_a = hpo_dict[edge[0]]
    hpo_b = hpo_dict[edge[1]]
    weight = 1.0  # TODO: CHANGE THIS IF YOU WANT TO ADD WEIGHTS TO HPO EDGES

    edge_index[0, i + len(combined_network)] = hpo_a
    edge_index[1, i + len(combined_network)] = hpo_b

    edge_weight[i + len(combined_network)] = weight

# save the edge_index and edge_weight tensors to pickle files
torch.save(edge_index, path.join('../data', 'edge_index.pt'))
torch.save(edge_weight, path.join('../data', 'edge_weight.pt'))
