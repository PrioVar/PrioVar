import torch
from typing import Tuple
from hpo import read_hpo_from_json, process_nodes, process_edges
from gene_mapping import get_gene_mapping_dict, get_combined_network, get_gene_phenotype_relations, get_gene_disease_relations
from disase_phenotype_mapping import proccess_hpoa
from os import path

def get_hpo_terms_edges() -> Tuple:
    graph_id, meta, nodes, edges, property_chain_axioms = read_hpo_from_json()
    items = process_nodes(nodes)

    # only create a list of id's from items
    hpo_list = [item['id'] for item in items]

    # sort the hpo_list
    hpo_list.sort()

    hpo_edges = process_edges(edges)

    return hpo_list, hpo_edges


combined_network = get_combined_network()
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


# disease
disease_phenotype_relations = proccess_hpoa()
disease_set = set()
for i, relation in enumerate(disease_phenotype_relations):
    disease_set.add(relation[0])

# print the number of elements in disease_set that starts with DECIPHER
#print(len([disease for disease in disease_set if disease.startswith('DECIPHER')]))
# 47 basıyor

disease_gene_relations = get_gene_disease_relations()
disease_set2 = set()
count = 0
for i, relation in enumerate(disease_gene_relations):
    # skip relation if the gene is not in the gene_mapping_dict
    if relation[0] not in gene_mapping_dict.keys():
        count += 1
        continue

    disease_set2.add(relation[1])

'''
# print number of genes (0. column in the relation) that is not in the gene_mapping_dict keys
print("Number of genes that is not in the gene_mapping_dict: ")
notInList = [relation[0] for relation in disease_gene_relations if relation[0] not in gene_dict.keys()]
print(len(notInList))
#print distinct in notInList
print("Number of distinct genes that is not in the gene_mapping_dict: ")
notInList = list(set(notInList))
# print the first 30 elements
print(notInList[:30])
print(notInList[30:60])
print(notInList[60:90])
print(notInList[90:])'''


disease_set = disease_set.union(disease_set2)

disease_list = list(disease_set)
disease_list.sort()

disease_dict = {}
for i in range(len(disease_list)):
    disease_dict[disease_list[i]] = i + len(gene_list) + len(hpo_list)

gene_phenotype_relations = get_gene_phenotype_relations()

'''
 Fill in other nodes
'''

# TODO: CHANGE THIS LINE IF YOU ADD MORE NODES
num_edges = (len(combined_network) + len(hpo_edges)
             + len(disease_phenotype_relations) + len(disease_gene_relations)
             + len(gene_phenotype_relations))

# create an edge_index tensor with size (2, num_edges)
edge_index = torch.zeros(2, num_edges)
edge_weight = torch.zeros(num_edges)

# fill in the network tensor
for i, row in enumerate(combined_network):
    ls = row.strip().split('\t')
    gene_a = gene_dict[gene_mapping_dict[ls[0]]]
    gene_b = gene_dict[gene_mapping_dict[ls[1]]]
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

for i, relation in enumerate(disease_phenotype_relations):
    disease = disease_dict[relation[0]]
    hpo = hpo_dict[relation[1]]
    weight = 1 #float(relation[2]) (şuan stringler olduğu için float'a çeviremiyoruz)

    edge_index[0, i + len(combined_network) + len(hpo_edges)] = disease
    edge_index[1, i + len(combined_network) + len(hpo_edges)] = hpo

    edge_weight[i + len(combined_network) + len(hpo_edges)] = weight

for i, relation in enumerate(disease_gene_relations):

    disease = disease_dict[relation[1]]
    gene = gene_dict[relation[0]]
    weight = 1

    edge_index[0, i + len(combined_network) + len(hpo_edges) + len(disease_phenotype_relations)] = disease
    edge_index[1, i + len(combined_network) + len(hpo_edges) + len(disease_phenotype_relations)] = gene

    edge_weight[i + len(combined_network) + len(hpo_edges) + len(disease_phenotype_relations)] = weight

for i, relation in enumerate(gene_phenotype_relations):

    gene = gene_dict[relation[0]]
    hpo = hpo_dict[relation[1]]
    weight = 1

    edge_index[0, i + len(combined_network) + len(hpo_edges) + len(disease_phenotype_relations) + len(disease_gene_relations)] = gene
    edge_index[1, i + len(combined_network) + len(hpo_edges) + len(disease_phenotype_relations) + len(disease_gene_relations)] = hpo

    edge_weight[i + len(combined_network) + len(hpo_edges) + len(disease_phenotype_relations) + len(disease_gene_relations)] = weight

# save the edge_index and edge_weight tensors to pickle files
torch.save(edge_index, path.join('../data', 'edge_index.pt'))
torch.save(edge_weight, path.join('../data', 'edge_weight.pt'))
