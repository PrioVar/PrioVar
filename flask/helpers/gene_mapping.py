import pandas as pd
from typing import Dict

def get_gene_mapping_dict() -> Dict[str, str]:
    # read data/identifier_mappings.txt into dataframe
    df = pd.read_csv("../data/identifier_mappings.txt", sep='\t', comment='#', header=None)

    # Collect all rows where 'Source' column is 'Gene Name'
    df = df[df[2] == 'Gene Name']

    # create a dictionary to map 'Preferred_Name' to 'Name'
    gene_dict = {}
    for index, row in df.iterrows():
        gene_dict[row[0]] = row[1]

    return gene_dict
