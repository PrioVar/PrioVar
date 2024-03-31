import torch
from typing import Tuple
from hpo import read_hpo_from_json, process_nodes, process_edges
from gene_mapping import get_gene_mapping_dict, get_combined_network, get_gene_phenotype_relations, get_gene_disease_relations
from disase_phenotype_mapping import process_hpoa
from os import path

# THIS IS A CONSTANT WHICH MIGHT BE CHANGED LATER
default_disease_phenotype_relation_frequency = 0.5


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

# handling diseases
disease_phenotype_relations, disease_database_ids = process_hpoa()

# create hash map for disease names. key -> database_id, value-> disease_name
disease_dict_db = {}
for i in range(len(disease_database_ids)):
    disease_name, database_ids = disease_database_ids[i]
    for database_id in database_ids:
        disease_dict_db[database_id] = disease_name

disease_set = set()
for i, relation in disease_phenotype_relations.iterrows():
    # add disease_name to the set
    disease_set.add(relation["disease_name"])


# disease_gene_relations are grouped by gene_symbol and aggregated disease_ids into a list
disease_gene_relations = get_gene_disease_relations()
disease_set2 = set()
disease_gene_relations_to_remove = []


for i, relation in enumerate(disease_gene_relations):
    # TODO: (change later) skip relation if the gene is not in the gene_mapping_dict
    if relation[0] not in gene_dict.keys():
        disease_gene_relations_to_remove.append(i)
        continue

    remove_ids = []
    for disease_id in relation[1]:

        if disease_id in disease_dict_db.keys():
            disease_set2.add(disease_dict_db[disease_id])
        else:
            remove_ids.append(disease_id)

    for remove_id in remove_ids:
        disease_gene_relations[i][1].remove(remove_id)


# removing the relations that are not in the gene_mapping_dict
disease_gene_relations = [relation for i, relation in enumerate(disease_gene_relations)
                          if i not in disease_gene_relations_to_remove]

disease_set = disease_set.union(disease_set2)
disease_list = list(disease_set)
disease_list.sort()

disease_dict = {}
for i in range(len(disease_list)):
    disease_dict[disease_list[i]] = i + len(gene_list) + len(hpo_list)

gene_phenotype_relations = get_gene_phenotype_relations()
gene_phenotype_relations_to_remove = []
for i, relation in enumerate(gene_phenotype_relations):
    # TODO: (change later) skip relation if the gene is not in the gene_mapping_dict
    if relation[0] not in gene_dict.keys():
        gene_phenotype_relations_to_remove.append(i)

gene_phenotype_relations = [relation for i, relation in enumerate(gene_phenotype_relations)
                            if i not in gene_phenotype_relations_to_remove]

'''
 Fill in other nodes
'''
NUM_NODES = len(gene_list) + len(hpo_list) + len(disease_list)

# write the number of nodes to a file
with open(path.join('../data', 'num_nodes.txt'), 'w') as file:
    file.write(str(NUM_NODES))

# TODO: CHANGE THIS PART IF YOU ADD MORE NODES/EDGES
num_edges = (len(combined_network) + len(hpo_edges)
             + len(gene_phenotype_relations))

for i, relation in disease_phenotype_relations.iterrows():
    num_edges += len(relation["hpo_id"])

for relation in disease_gene_relations:
    if relation[0] not in gene_dict.keys():
        continue
    num_edges += len(relation[1])


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


disease_phenotype_count = 0
for i, relation in disease_phenotype_relations.iterrows():
    disease = disease_dict[relation["disease_name"]]
    hpo_id_list = relation["hpo_id"]
    frequency = relation["frequency"]

    for j in range(len(relation["hpo_id"])):
        weight = frequency[j]
        hpo = hpo_dict[hpo_id_list[j]]

        edge_index[0, disease_phenotype_count + len(combined_network) + len(hpo_edges)] = disease
        edge_index[1, disease_phenotype_count + len(combined_network) + len(hpo_edges)] = hpo

        edge_weight[disease_phenotype_count + len(combined_network) + len(hpo_edges)] = weight
        disease_phenotype_count += 1


disease_gene_count = 0
for i, relation in enumerate(disease_gene_relations):
    # TODO: maybe fix this later
    if relation[0] not in gene_dict.keys():
        continue

    # for loop needed because there can be multiple diseases for a gene
    for disease_id in relation[1]:
        if disease_id not in disease_dict_db.keys():
            continue

        disease = disease_dict[disease_dict_db[disease_id]]
        gene = gene_dict[relation[0]]
        weight = 1.0

        edge_index[0, disease_gene_count + len(combined_network) + len(hpo_edges) + disease_phenotype_count] = disease
        edge_index[1, disease_gene_count + len(combined_network) + len(hpo_edges) + disease_phenotype_count] = gene

        edge_weight[disease_gene_count + len(combined_network) + len(hpo_edges) + disease_phenotype_count] = weight

        disease_gene_count += 1


for i, relation in enumerate(gene_phenotype_relations):
    # TODO: maybe fix this later
    if relation[0] not in gene_dict.keys():
        continue

    gene = gene_dict[relation[0]]
    hpo = hpo_dict[relation[1]]
    weight = 1.0

    edge_index[0, i + len(combined_network) + len(hpo_edges) + disease_phenotype_count + disease_gene_count] = gene
    edge_index[1, i + len(combined_network) + len(hpo_edges) + disease_phenotype_count + disease_gene_count] = hpo

    edge_weight[i + len(combined_network) + len(hpo_edges) + disease_phenotype_count + disease_gene_count] = weight

# save the edge_index and edge_weight tensors to pickle files
torch.save(edge_index, path.join('../data', 'edge_index.pt'))
torch.save(edge_weight, path.join('../data', 'edge_weight.pt'))

# save the gene, hpo, and disease dictionaries to pickle files
torch.save(gene_dict, path.join('../data', 'gene_dict.pt'))
torch.save(hpo_dict, path.join('../data', 'hpo_dict.pt'))
torch.save(disease_dict, path.join('../data', 'disease_dict.pt'))

