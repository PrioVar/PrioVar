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

    df1 = df[df["frequency"] == '-']
    print(len(df1))

    # TODO: handle those with '-' by replacing with 1/2???
    # TODO: handle rows with % sign in frequency column
    # TODO: handle rows with HPO id in frequency column by replacing with 1/5, 2/5, 3/5, 4/5, 5/5

    # return the list of gene-phenotype relations by getting only gene_symbol and hpo_id columns
    return df[["gene_symbol", "hpo_id"]].values.tolist()

def get_gene_phenotype_relations_and_frequency() -> List[List]:
    """
    :return: List of gene-phenotype relations. Each relation is a list of two
    elements: gene_symbol (HPO format) and hpo_id and frequency
    example: [['A1BG', 1234567], ['B3GALT6', 15]]
    """

    # read ../data/genes_to_phenotype.txt into dataframe
    df = pd.read_csv("../data/genes_to_phenotype.txt", sep='\t', comment='#')

    # process hpo_id column to get the last 7 characters and convert to int
    df["hpo_id"] = df["hpo_id"].str[-7:].astype(int)

    def convert_to_float(x):
        if '/' in x:
            return (float(x.split('/')[0]) + 1) / (float(x.split('/')[1]) + 2)
        elif "%" in x:
            return float(x.split('%')[0])/100
        elif x == 'HP:0040285':
            return 0.0
        elif x == 'HP:0040284':
            return 0.025
        elif x == 'HP:0040282':
            return 0.55
        elif x == 'HP:0040283':
            return 0.17
        elif x == 'HP:0040281':
            return 0.9
        else:
            return None


    df["frequency"] = df["frequency"].apply(lambda x: convert_to_float(str(x)))


    # return the list of gene-phenotype relations by getting only gene_symbol and hpo_id columns
    return df[["gene_symbol", "hpo_id", "frequency"]].values.tolist()



def get_gene_disease_relations() -> List[List]:
    """
    :return: List of gene-disease relations
    example: [ ['A2ML1', ['OMIM:615120','ORPHA:12354']], ['A4GALT', ['OMIM:6151240','ORPHA:123254']]]
    """

    # read ../data/genes_to_disease.txt into dataframe
    df = pd.read_csv("../data/genes_to_disease.txt", sep='\t', comment='#')

    # Delete rows with gene='-'
    df = df[df['gene_symbol'] != '-']

    # Group by gene_symbol and aggregate disease_ids into a list
    grouped = df.groupby('gene_symbol')['disease_id'].agg(list).reset_index()

    # Sort disease_ids lists based on the order 'OMIM', 'ORPHA', 'DECIPHER'
    grouped['disease_id'] = grouped['disease_id'].apply(custom_sort)

    # Convert DataFrame to list of tuples
    gene_disease_relations = grouped.apply(lambda x: (x['gene_symbol'], x['disease_id']), axis=1).tolist()

    return gene_disease_relations


def custom_sort(disease_ids):
    order = {'OMIM': 0, 'ORPHA': 1, 'DECIPHER': 2}
    return sorted(disease_ids, key=lambda x: order.get(x.split(':')[0], float('inf')))


# deprecated get_gene_phenotype_relations
"""
def get_gene_disease_relations() -> List[List]:
    
    :return: List of gene-disease relations
    example: [['A1BG', 'OMIM:615120'], ['B3GALT6', 'OMIM:615120']]
    

    # read ../data/genes_to_disease.txt into dataframe
    df = pd.read_csv("../data/genes_to_disease.txt", sep='\t', comment='#')

    # TODO: delete rows with gene='-'

    # return the list of gene-disease relations by getting only gene_symbol and disease_id columns
    return df[["gene_symbol", "disease_id"]].values.tolist()
"""
