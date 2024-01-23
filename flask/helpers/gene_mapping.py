import pandas as pd
from os import path
from typing import Dict, List

def get_combined_network():
    # read the combined genemania network into a tensor
    # in each row, there's Gene_A, Gene_B, and the edge weight
    with open(path.join('../data', 'COMBINED.DEFAULT_NETWORKS.BP_COMBINING.txt'), 'r') as f:
        combined_network = f.readlines()

    # remove the header
    combined_network = combined_network[1:]

    return combined_network

def get_gene_mapping_dict() -> Dict[str, str]:
    """
    :return: Dictionary that maps Genemanie gene names to HPO gene names
    example: {'P48506': 'A1BG', 'O60762': 'A1CF', ...}
    """
    # read data/identifier_mappings.txt into dataframe
    df = pd.read_csv("../data/identifier_mappings.txt", sep='\t', comment='#')

    # Collect all rows where 'Source' column is 'Gene Name'
    df = df[df['Source'] == 'Gene Name']

    # create a dictionary to map 'Preferred_Name' to 'Name'
    gene_dict = {}
    for index, row in df.iterrows():
        gene_dict[row[0]] = row[1]

    return gene_dict


def get_gene_phenotype_relations() -> List[List]:
    """
    :return: List of gene-phenotype relations. Each relation is a list of two
    elements: gene_symbol (HPO format) and hpo_id
    example: [['A1BG', 1234567], ['B3GALT6', 15]]
    """

    # read ../data/genes_to_phenotype.txt into dataframe
    df = pd.read_csv("../data/genes_to_phenotype.txt", sep='\t', comment='#')

    # process hpo_id column to get the last 7 characters and convert to int
    df["hpo_id"] = df["hpo_id"].str[-7:].astype(int)

    # return the list of gene-phenotype relations by getting only gene_symbol and hpo_id columns
    return df[["gene_symbol", "hpo_id"]].values.tolist()


def get_gene_disease_relations() -> List[List]:
    """
    :return: List of gene-disease relations
    example: [['A1BG', 'OMIM:615120'], ['B3GALT6', 'OMIM:615120']]
    """

    # read ../data/genes_to_disease.txt into dataframe
    df = pd.read_csv("../data/genes_to_disease.txt", sep='\t', comment='#')

    # return the list of gene-disease relations by getting only gene_symbol and disease_id columns
    return df[["gene_symbol", "disease_id"]].values.tolist()