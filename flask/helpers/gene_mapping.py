import pandas as pd
from typing import Dict, List

def get_gene_mapping_dict() -> Dict[str, str]:
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
    :return: List of gene-phenotype relations
    """

    # read ../data/genes_to_phenotype.txt into dataframe
    df = pd.read_csv("../data/genes_to_phenotype.txt", sep='\t', comment='#')

    # process hpo_id column to get the last 7 characters and convert to int
    df["hpo_id"] = df["hpo_id"].str[-7:].astype(int)

    # return the list of gene-phenotype relations by getting only gene_symbol and hpo_id columns
    return df[["gene_symbol", "hpo_id"]].values.tolist()
