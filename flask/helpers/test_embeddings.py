from train1 import *
from hpo import read_hpo_from_json, process_nodes, process_edges


def get_hpo_terms_edges():
    graph_id, meta, nodes, edges, property_chain_axioms = read_hpo_from_json()
    items = process_nodes(nodes)

    # only create a list of id's from items
    hpo_list = [item['id'] for item in items]

    # sort the hpo_list
    hpo_list.sort()

    hpo_edges = process_edges(edges)

    return hpo_list, hpo_edges


path_embedding = '../data/node_embeddings.txt'

path_gene_dict = '../data/gene_dict.pt'
path_hpo_dict = '../data/hpo_dict.pt'

# read the embeddings from the file using the read_embeddings function
embeddings = read_embedding('../data/node_embeddings.txt')

# read dictionaries from the file using the read_dict function
gene_dict, hpo_dict = read_dicts(path_gene_dict, path_hpo_dict)

hpo_list, hpo_edges = get_hpo_terms_edges()  # get the hpo terms and edges


# for the hpo term with id "1", get the embedding
def get_hpo_embedding(hpo_id: int):
    return embeddings[hpo_dict[hpo_id]]


def get_gene_embedding(gene_name: str):
    return embeddings[gene_dict[gene_name]]


# in this Node2Vec embedding, get the closest hpo term to the hpo term with id "1"
def get_closest_hpo(hpo_id: int):
    hpo_embedding = get_hpo_embedding(hpo_id)
    closest_hpo_id = None
    closest_hpo_index = None
    min_distance = np.inf

    for id in hpo_list:
        if id == hpo_id:
            continue
        embedding = get_hpo_embedding(id)
        distance = np.linalg.norm(hpo_embedding - embedding)
        if distance < min_distance:
            min_distance = distance
            closest_hpo_id = id
            closest_hpo_index = hpo_dict[id]

    return closest_hpo_id, closest_hpo_index, min_distance


#closest1, index1, distance1 = get_closest_hpo(1)
#print(f"Closest HPO term to HPO term with id 1 is HPO term with id {closest1} with index {index1} and distance {distance1}")



