import torch
from os import path

# read the combined genemania network into a tensor
# in each row, there's Gene_A, Gene_B, and the edge weight
with open(path.join('data', 'COMBINED.DEFAULT_NETWORKS.BP_COMBINING.txt'), 'r') as f:
    combined_network = f.readlines()

# remove the header
combined_network = combined_network[1:]

# TODO: REMOVE THIS
combined_network = combined_network[:10]

gene_set = set()
for row in combined_network:
    ls = row.strip().split('\t')
    gene_set.add(ls[0])
    gene_set.add(ls[1])

gene_list = list(gene_set)
gene_list.sort()

# create a dictionary to map gene names to indices
node_dict = {}
for i in range(len(gene_list)):
    node_dict[gene_list[i]] = i

'''
 Fill in other nodes
'''

# TODO: CHANGE THIS LINE IF YOU ADD MORE NODES
num_edges = len(combined_network)

# create an edge_index tensor with size (2, num_edges)
edge_index = torch.zeros(2, num_edges)
edge_weight = torch.zeros(num_edges)

# fill in the network tensor
for i, row in enumerate(combined_network):
    ls = row.strip().split('\t')
    gene_a = node_dict[ls[0]]
    gene_b = node_dict[ls[1]]
    weight = float(ls[2])

    edge_index[0, i] = gene_a
    edge_index[1, i] = gene_b

    edge_weight[i] = weight

# save the edge_index and edge_weight tensors to pickle files
torch.save(edge_index, path.join('data', 'edge_index.pt'))
torch.save(edge_weight, path.join('data', 'edge_weight.pt'))



