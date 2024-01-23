# proccess phenotype.hpoa
#example row:
# database_id	disease_name	qualifier	hpo_id	reference	evidence	onset	frequency	sex	modifier	aspect	biocuration
# OMIM:619340	Developmental and epileptic encephalopathy 96		HP:0011097	PMID:31675180	PCS		1/2			P	HPO:probinson[2021-06-21]
import csv
from os import path
import pandas as pd
#proccess genes_to_phenotype.txt

# get database_id, hpo_id, frequency

# store in list of tuples

# start reading phenotype.hpoa
eskikod = """# read phenotype.hpoa line by line and find and find "Omim: " and "HP: " and store in list of tuples
    list_of_tuples = []
    with open(path.join('../data', 'phenotype.hpoa'), 'r') as file:
        line_index = 1
        for line in file:

            # skip the first 5 lines
            if line_index <= 5:
                line_index += 1
                continue
            line_index += 1

            disase_id = ""
            hpo_id = ""

            words = line.split()
            for word in words:
                if word.startswith("OMIM:") or word.startswith("ORPHA:"):
                    disase_id = word
                elif word.startswith("HP:"):
                    hpo_id = word

            if disase_id != "" and hpo_id != "":
                #convert hpo_id to int
                hpo_id = int(hpo_id[-7:])
                list_of_tuples.append((disase_id, hpo_id))

    return list_of_tuples"""
def proccess_hpoa():

    # read phenotype.hpoa skip the first 4 lines and store in dataframe

    df = pd.read_csv(path.join('../data', 'phenotype.hpoa'), sep='\t', comment='#', skiprows=4)

    # get database_id, hpo_id, frequency
    df = df[["database_id", "hpo_id", "frequency"]]

    # rteurn list of tuples
    return df.values.tolist()




list_of_tuples = proccess_hpoa()

