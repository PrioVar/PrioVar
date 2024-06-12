import os
import pickle
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import torch
from helpers import hpo_sample
from helpers.gene_mapping import get_gene_phenotype_relations, get_gene_phenotype_relations_and_frequency

path_variants = '../data/5bc6f943-66e0-4254-94f5-ed3888f05d0a.vep.tsv'
path_embedding = '../data/node_embeddings.txt'

path_gene_dict = '../data/gene_dict.pt'
path_hpo_dict = '../data/hpo_dict.pt'

# triplets of strategies for hpo_sample.sample_from_random_strategy like (precise, imprecise, noisy)
HPO_SAMPLE_STRATEGIES = [(3, 2, 1), (4, 2, 1), (2, 1, 0), (1, 2, 0), (2, 2, 0)]

neutral_embedding = None


def read_variants():
    # read by skipping 48 lines
    df = pd.read_csv(path_variants, sep='\t', skiprows=48)
    return df


def read_embedding(path_of_embedding=path_embedding):
    node_embeddings = np.loadtxt(path_of_embedding, skiprows=1)

    # sort the array by the first column
    node_embeddings = node_embeddings[node_embeddings[:, 0].argsort()]

    # remove the first column
    node_embeddings = node_embeddings[:, 1:]
    return node_embeddings


def calculate_neutral_embedding_as_root_embedding(node_embeddings, hpo_dict):
    # get the embedding of first phenotype
    root_embedding = node_embeddings[hpo_dict[1]]
    return root_embedding


def read_dicts(gene_path=path_gene_dict, hpo_path=path_hpo_dict):
    gene_dict = torch.load(gene_path)
    hpo_dict = torch.load(hpo_path)
    return gene_dict, hpo_dict


important_noticeIII_some_genes_in_the_annotation_is_not_in_graph = """# check if symbols are in gene_dict
not_in_gene_dict = []
for symbol in symbols:
    if symbol not in gene_dict:
        not_in_gene_dict.append(symbol)

print(not_in_gene_dict)
print(len(not_in_gene_dict))
#print(gene_dict.keys())
 159817 not in the gene_dict
4029961 in the gene dict
"""


def add_sampled_hpo_terms_full_precise(
    df_variants,
    node_embeddings,
    gene_dict,
    hpo_dict,
    number_of_phenotypes
):
    gene_phenotype_relations = get_gene_phenotype_relations()
    global neutral_embedding
    # for each gene in the dataframe, sample number_of_phenotypes HPO terms
    for i, row in df_variants.iterrows():
        gene = row['SYMBOL']

        # if the gene is not in the gene_dict, only the global neutral_embedding will be added
        if gene not in gene_dict:
            for j in range(number_of_phenotypes):
                df_variants.at[i, f'embedding{j}'] = neutral_embedding
            continue

        # get the HPO terms for the gene
        hpo_terms = [relation[1] for relation in gene_phenotype_relations if relation[0] == gene]

        # sample number_of_phenotypes HPO terms (if there are less than number_of_phenotypes, sample global neutral_embedding for the rest)
        sampled_hpo = random.sample(hpo_terms, min(len(hpo_terms), number_of_phenotypes))
        sampled_embedding = [node_embeddings[hpo_dict[hpo]] for hpo in sampled_hpo]

        # if there are less than number_of_phenotypes, sample global neutral_embedding for the rest
        if len(sampled_hpo) < number_of_phenotypes:
            sampled_embedding.extend([neutral_embedding] * (number_of_phenotypes - len(sampled_hpo)))

        # add the sampled embeddings to the dataframe
        for j in range(number_of_phenotypes):
            df_variants.at[i, f'embedding{j}'] = sampled_embedding[j]

    return df_variants


"""
if False:
    # read the variants
    df_variants = read_variants()

    # show class distribution for CLIN_SIG column
    dist = df_variants['CLIN_SIG'].value_counts()

    # only get the rows with pathogenic or benign, likely_benign, likely_pathogenic values in CLIN_SIG column

    print("Ratio of pathogenic to benign: ", len(df_variants[df_variants['CLIN_SIG'] == 'pathogenic']) / len(df_variants))
    print("size of df_variants: ", df_variants.shape)

    # read the node embeddings
    node_embeddings = read_embedding()

    # read the gene and hpo dictionaries
    gene_dict, hpo_dict = read_dicts()

    # calculate the neutral embedding as the root embedding
    neutral_embedding = calculate_neutral_embedding_as_root_embedding(node_embeddings, hpo_dict)

    # get the types of the columns
    print(df_variants.dtypes)
    # get the first row and print everything's type
    print(df_variants.iloc[0].apply(type))

    #len rows
    print(len(df_variants))

"""


#example row
#Uploaded_variation	Allele	Gene	Feature	Consequence	Existing_variation	SYMBOL	CANONICAL	SIFT	PolyPhen	HGVSc	HGVSp	AF	gnomADe_AF	CLIN_SIG	REVEL	SpliceAI_pred	DANN_score	MetaLR_score	CADD_raw_rankscore	ExAC_AF	ALFA_Total_AF	AlphaMissense_score	AlphaMissense_pred	turkishvariome_TV_AF
#2205837	G	ENSG00000186092	ENST00000641515	missense_variant	rs781394307	OR4F5	YES	tolerated(0.08)	benign(0.007)	ENST00000641515.2:c.107A>G	ENSP00000493376.2:p.Glu36Gly	-	0.02667	likely_benign	0.075	OR4F5|0.00|0.00|0.02|0.03|45|-1|-19|-1	0.95777141322514492	0.0013	0.24798	1.655e-03	0.0011802394199966278	0.0848,0.0854	B,B	-

## Column descriptions:
## Uploaded_variation : Identifier of uploaded variant
## Allele : The variant allele used to calculate the consequence
## Gene : Stable ID of affected gene
## Feature : Stable ID of feature
## Consequence : Consequence type
## Existing_variation : Identifier(s) of co-located known variants
## SYMBOL : Gene symbol (e.g. HGNC)
## CANONICAL : Indicates if transcript is canonical for this gene
## SIFT : SIFT prediction and/or score
## PolyPhen : PolyPhen prediction and/or score
## HGVSc : HGVS coding sequence name
## HGVSp : HGVS protein sequence name
## AF : Frequency of existing variant in 1000 Genomes combined population
## gnomADe_AF : Frequency of existing variant in gnomAD exomes combined population
## CLIN_SIG : ClinVar clinical significance of the dbSNP variant
## REVEL : Rare Exome Variant Ensemble Learner
## SpliceAI_pred : SpliceAI predicted effect on splicing. These include delta scores (DS) and delta positions (DP) for acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). Format: SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL
## DANN_score : (from dbNSFP4.5a_grch38) DANN is a functional prediction score retrained based on the training data of CADD using deep neural network. Scores range from 0 to 1. A larger number indicate a higher probability to be damaging. More information of this score can be found in doi: 10.1093/bioinformatics/btu703.
## MetaLR_score : (from dbNSFP4.5a_grch38) Our logistic regression (LR) based ensemble prediction score, which incorporated 10 scores (SIFT, PolyPhen-2 HDIV, PolyPhen-2 HVAR, GERP++, MutationTaster, Mutation Assessor, FATHMM, LRT, SiPhy, PhyloP) and the maximum frequency observed in the 1000 genomes populations. Larger value means the SNV is more likely to be damaging. Scores range from 0 to 1.
## CADD_raw_rankscore : (from dbNSFP4.5a_grch38) CADD raw scores were ranked among all CADD raw scores in dbNSFP. The rankscore is the ratio of the rank of the score over the total number of CADD raw scores in dbNSFP. Please note the following copyright statement for CADD: "CADD scores (http://cadd.gs.washington.edu/) are Copyright 2013 University of Washington and Hudson-Alpha Institute for Biotechnology (all rights reserved) but are freely available for all academic, non-commercial applications. For commercial licensing information contact Jennifer McCullar (mccullaj@uw.edu)."
## ExAC_AF : Frequency of existing variant in ExAC combined population
## ALFA_Total_AF : (from dbNSFP4.5a_grch38) Alternative allele frequency of the total samples in the Allele Frequency Aggregator
## AlphaMissense_score : (from dbNSFP4.5a_grch38) AlphaMissense is a unsupervised model for predicting the pathogenicity of human missense variants by incorporating structural context of an AlphaFold-derived system. The AlphaMissense score ranges from 0 to 1. The larger the score, the more likely the variant is pathogenic. Detals see https://doi.org/10.1126/science.adg7492. License information: "AlphaMissense Database Copyright (2023) DeepMind Technologies Limited. All predictions are provided for non-commercial research use only under CC BY-NC-SA license." This distribution of AlphaMissense_score, AlphaMissense_rankscore, and AlphaMissense_pred are also under CC BY-NC-SA license. A copy of CC BY-NC-SA license can be found at https://creativecommons.org/licenses/by-nc-sa/4.0/.
## AlphaMissense_pred : (from dbNSFP4.5a_grch38) The AlphaMissense classification of likely (B)enign, (A)mbiguous, or likely (P)athogenic with 90% expected precision estimated from ClinVar for likely benign and likely pathogenic classes.
## turkishvariome_TV_AF : TV_AF field from [PATH]/TurkishVariome.vcf.gz
def clean_data(
    df_variants,
    output_pickle_file='../data/variants_cleaned.pkl',
    labels=['pathogenic', 'benign', 'likely_benign', 'likely_pathogenic']
):
    # only get the rows with pathogenic or benign, likely_benign, likely_pathogenic values in CLIN_SIG column
    df_variants = df_variants[df_variants['CLIN_SIG'].isin(labels)]

    # divide  HGVSc  columns into two columns by splitting the
    # values by ':' ENST00000616125.5:c.11G>A
    df_variants[['HGVSc', 'HGVSc2']] = df_variants['HGVSc'].str.split(':', expand=True)

    # extract the info from the HGVSc2 column so that we have the 2
    # more columns 11, G>A # ENST00000616125.5:c.11G>A
    df_variants[['HGVSc_number', 'HGVSc_change']] = df_variants['HGVSc2'].str.extract(r'c\.(\d+)([A-Z]>.*)')

    # divide HGVSp columns into two columns by splitting the
    # values by ':' ENSP00000484643.1:p.Gly4Glu
    df_variants[['HGVSp', 'HGVSp2']] = df_variants['HGVSp'].str.split(':', expand=True)

    # extract the info from the HGVSp2 column so that we have the 2 more columns 4,
    # GlyGlu namely number in the middle and the change
    # pay attention that there should be 2 columns
    df_variants[['HGVSp_from', 'HGVSp_number', 'HGVSp_to']] = df_variants['HGVSp2'].str.extract(
        r'p\.([A-Z][a-z]+)(\d+)([A-Z][a-z]+)')

    # combine from and to columns to get the change
    df_variants['HGVSp_change'] = df_variants['HGVSp_from'] + df_variants['HGVSp_to']

    # drop the columns that are not needed
    df_variants = df_variants.drop(columns=['HGVSc', 'HGVSc2', 'HGVSp2', 'HGVSp_from', 'HGVSp_to'])

    # divide  SpliceAI_pred  columns into 8 columns by
    # splitting the values by '|' and drop the first column
    df_variants[['SpliceAI_pred_symbol', 'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL']] \
        = df_variants['SpliceAI_pred'].str.split('|', expand=True)
    df_variants = df_variants.drop(columns=['SpliceAI_pred', 'SpliceAI_pred_symbol'])

    # alpha missense scorelarda . lar var
    # eliminate ,., values for example 0.0848,.,0.0854 -> 0.0848,0.0854 or .,0.5 -> 0.5
    # or 0.5,. -> 0.5 so, replace . , pairs that is adjacent to each other
    df_variants[['AlphaMissense_score_mean', 'AlphaMissense_std_dev']] = df_variants['AlphaMissense_score'].apply(
        lambda x: (np.nan, np.nan) if (x == "-" or x == "") else
        # Filter the split results to include only valid floats, then perform calculations
        ((numbers := np.array([float(num) for num in x.split(',') if num.replace('.', '', 1).isdigit()]),
          np.mean(numbers), np.std(numbers))[1:]
         if any(num.replace('.', '', 1).isdigit() for num in x.split(','))
         else (np.nan, np.nan))
    ).apply(pd.Series)

    # alpha missense pred has several values ("A", "B", "P") seperated by commas,
    # change this column to ratio of letters (pay attention to "-" values)
    df_variants['AlphaMissense_pred'] = df_variants['AlphaMissense_pred'].apply(lambda x: (x.count('A'), x.count('B'), x.count('P')) if x != "-" else (np.nan, np.nan, np.nan))

    # divide it so that we have 3 columns for each letter storing the ratio of
    # the letter in the column if 3 A's 2 B's and 1 P's then 3/6, 2/6, 1/6
    df_variants[['AlphaMissense_pred_A', 'AlphaMissense_pred_B', 'AlphaMissense_pred_P']] \
        = df_variants['AlphaMissense_pred'].apply(
        lambda x: (np.nan, np.nan, np.nan) if x == "-" or sum(x) == 0 else (x[0] / sum(x), x[1] / sum(x), x[2] / sum(x))).apply(pd.Series)

    # HGSV.. columns
    # make the column HGVSc_number a numerical column for model (pay attention to "-" values)
    df_variants['HGVSc_number'] = df_variants['HGVSc_number'].replace('-', np.nan).astype('float')

    # make the column HGVSp_number a numerical column for model (pay attention to "-" values)
    df_variants['HGVSp_number'] = df_variants['HGVSp_number'].replace('-', np.nan).astype('float')

    # make the column HGVSp_change a categorical column for model
    df_variants['HGVSp_change'] = df_variants['HGVSp_change'].astype('category')

    # make the column HGVSc_change a categorical column for model
    df_variants['HGVSc_change'] = df_variants['HGVSc_change'].astype('category')

    # make the column canonical a categorical column for model
    df_variants['CANONICAL'] = df_variants['CANONICAL'].astype('category')

    # make the Allele column a categorical column for model
    df_variants['Allele'] = df_variants['Allele'].astype('category')

    # make the column polyphen a numerical column and categorical for model by
    # extracting the number from the string and taking the string before by looking for the '(number)'
    df_variants['PolyPhen_number'] = df_variants['PolyPhen'].str.extract(r'\((\d+\.\d+)\)').astype('float')
    df_variants['PolyPhen'] = df_variants['PolyPhen'].str.extract(r'(\w+)').astype('category')

    # make the column SIFT a numerical column and categorical for model by
    # extracting the number from the string and taking the string before by looking for the '(number)'
    df_variants['SIFT_number'] = df_variants['SIFT'].str.extract(r'\((\d+\.\d+)\)').astype('float')
    df_variants['SIFT'] = df_variants['SIFT'].str.extract(r'(\w+)').astype('category')

    # make the gene column a categorical column for model
    df_variants['SYMBOL'] = df_variants['SYMBOL'].astype('category')

    # make the column Feature a categorical column for model
    df_variants['Feature'] = df_variants['Feature'].astype('category')

    # make the column Consequence a categorical column for model
    df_variants['Consequence'] = df_variants['Consequence'].astype('category')

    # make the column Existing_variation (values are seperated by commas) a categorical column for model by splitting the values
    #df_variants['Existing_variation'] = df_variants['Existing_variation'].str.split(',').explode().astype('category')

    # make the column AF a numerical column for model (pay attention to "-" values)
    df_variants['AF'] = df_variants['AF'].replace('-', np.nan).astype('float')

    # make the column gnomADe_AF a numerical column for model (pay attention to "-" values)
    df_variants['gnomADe_AF'] = df_variants['gnomADe_AF'].replace('-', np.nan).astype('float')

    # make the column CLIN_SIG a categorical column for model
    df_variants['CLIN_SIG'] = df_variants['CLIN_SIG'].astype('category')

    # make the column REVEL a numerical column for model (pay attention to "-" values)
    df_variants['REVEL'] = df_variants['REVEL'].replace('-', np.nan).astype('float')

    # make the column DANN_score a numerical column for model (pay attention to "-" values)
    df_variants['DANN_score'] = df_variants['DANN_score'].replace('-', np.nan).astype('float')

    # make the column MetaLR_score a numerical column for model (pay attention to "-" values)
    df_variants['MetaLR_score'] = df_variants['MetaLR_score'].replace('-', np.nan).astype('float')

    # make the column CADD_raw_rankscore a numerical column for model (pay attention to "-" values)
    df_variants['CADD_raw_rankscore'] = df_variants['CADD_raw_rankscore'].replace('-', np.nan).astype('float')

    # make the column ExAC_AF a numerical column for model (pay attention to "-" values)
    df_variants['ExAC_AF'] = df_variants['ExAC_AF'].replace('-', np.nan).astype('float')

    # make the column ALFA_Total_AF a numerical column for model (pay attention to "-" values)
    df_variants['ALFA_Total_AF'] = df_variants['ALFA_Total_AF'].replace('-', np.nan).astype('float')

    # make the column turkishvariome_TV_AF a numerical column for model (pay attention to "-" values)
    df_variants['turkishvariome_TV_AF'] = df_variants['turkishvariome_TV_AF'].replace('-', np.nan).astype('float')

    # save the dataframe to a new file that preserves data types (category, numerical etc. for each column)
    # pickle the dataframe
    df_variants.to_pickle(output_pickle_file)


# call
#df_variants = read_variants()
# output to data folder
#clean_data(df_variants, '../data/variants_cleaned.pkl')
#exit()


def sample_hpo_terms_for_variants_optimized(
    df_variants,
    gene_dict,
    max_ancestral_depth=10,
    put_in_the_df=False
):

    # order df_variants by SYMBOL
    df_variants = df_variants.sort_values(by='SYMBOL')

    # add new column to the dataframe to store the sampled HPO terms
    if put_in_the_df:
        # np.nan is used to represent missing values
        df_variants['sampled_hpo'] = np.nan

    # get the gene-phenotype relations
    gene_phenotype_relations = get_gene_phenotype_relations()

    # create a network object from hpo_sample
    network = hpo_sample.Network()

    # create dictionary named: sampled_hpoIDs_for_variants
    # key: variant name (e.g. 'rs12345'), value: list of sampled HPO terms
    sampled_hpoIDs_for_variants = {}

    # processed variants
    count = 0
    j = 0
    print("Loop for phenotypes started")
    # for each gene in the dataframe (ordered) , sample hpo terms using get_pool function
    # don't calculate the hpo terms for the same gene again
    last_gene = None
    last_hpo_terms = []
    last_hpo_pool = []
    for i, row in df_variants.iterrows():
        if j % 1000 == 0:
            print("Row: ", j)
        j += 1

        # get the gene
        gene = row['SYMBOL']

        # if the gene is not in the gene_dict, continue to the next iteration
        if gene not in gene_dict:
            continue

        # get the HPO terms for the gene from the gene-phenotype relations
        if gene != last_gene:
            hpo_terms = [relation[1] for relation in gene_phenotype_relations if relation[0] == gene]
            last_gene = gene
            last_hpo_terms = hpo_terms
            # TODO: A problem might occur here if both parents of a term are in the list
            last_hpo_pool = network.get_imprecision_pool(hpo_terms, max_ancestral_depth)

        if len(last_hpo_terms) == 0:
            continue

        count += 1
        if count % 100 == 0:
            print("Processed variants (sample pheno) : ", count)

        strategy = random.choice(HPO_SAMPLE_STRATEGIES)
        precision = min(strategy[0], len(last_hpo_terms))
        imprecision = min(strategy[1], len(last_hpo_pool))

        precise_samples = random.sample(last_hpo_terms, precision)
        imprecise_samples = random.sample(last_hpo_pool, imprecision)

        combined = precise_samples + imprecise_samples

        # no noisy for now

        # add the sampled HPO terms to the dictionary
        sampled_hpoIDs_for_variants[row['#Uploaded_variation']] = combined

        if put_in_the_df:
            df_variants.at[i, 'sampled_hpo'] = ','.join([str(hpo) for hpo in combined])

    # save dictionary to a file
    with open('sampled_hpoIDs_for_variants.pkl', 'wb') as f:
        pickle.dump(sampled_hpoIDs_for_variants, f)

    print("Sampled HPO terms for variants are saved to sampled_hpoIDs_for_variants.pkl")
    return sampled_hpoIDs_for_variants, df_variants


def sample_hpo_terms_with_frequency_optimized(
    df_variants,
    gene_dict,
    output_pickle_file='../data/sampled_hpoIDs_with_freq_for_variants.pkl',
    max_ancestral_depth=10,
    put_in_the_df=False
):
    # order df_variants by SYMBOL
    df_variants = df_variants.sort_values(by='SYMBOL')

    # add new column to the dataframe to store the sampled HPO terms
    if put_in_the_df:
        # np.nan is used to represent missing values
        df_variants['sampled_hpo'] = np.nan

    # get the gene-phenotype relations
    gene_phenotype_relations = get_gene_phenotype_relations_and_frequency()

    # create a network object from hpo_sample
    network = hpo_sample.Network()

    # create dictionary named: sampled_hpoIDs_for_variants
    # key: variant name (e.g. 'rs12345'), value: list of sampled HPO terms
    sampled_hpoIDs_for_variants = {}

    # processed variants
    count = 0
    j = 0
    print("Loop for phenotypes started")
    # for each gene in the dataframe (ordered) , sample hpo terms using get_pool function
    # don't calculate the hpo terms for the same gene again
    last_gene = None
    last_hpo_terms = []
    last_hpo_pool = []
    # among HPO_SAMPLE_STRATEGIES get the max of sum of precise and imprecise not noisy
    max_number_of_relevant_phenotypes = max([sum(strategy[:2]) for strategy in HPO_SAMPLE_STRATEGIES])
    for i, row in df_variants.iterrows():
        if j % 1000 == 0:
            print("Row: ", j)
        j += 1

        # get the gene
        gene = row['SYMBOL']

        # if the gene is not in the gene_dict, continue to the next iteration
        if gene not in gene_dict:
            continue

        # get the HPO terms for the gene from the gene-phenotype relations
        if gene != last_gene:
            hpo_terms_and_frequencies = [relation[1:] for relation in gene_phenotype_relations if relation[0] == gene]

            # get the terms with the highest frequencies (top 10) pay attention to None values consider them as 0
            hpo_terms_and_frequencies = sorted(hpo_terms_and_frequencies,
                                               key=lambda x: x[1] if x[1] is not None else 0,
                                               reverse=True)

            # get the most frequent max_number_of_relevant_phenotypes HPO terms
            # if there are less than max_number_of_relevant_phenotypes, get all
            hpo_terms = [relation[0]
                         for relation in
                         hpo_terms_and_frequencies[:max_number_of_relevant_phenotypes]]

            last_gene = gene
            last_hpo_terms = hpo_terms
            # TODO: A problem might occur here if both parents of a term are in the list
            last_hpo_pool = network.get_imprecision_pool(hpo_terms, max_ancestral_depth)

        if len(last_hpo_terms) == 0:
            continue

        count += 1
        if count % 100 == 0:
            print("Processed variants (sample pheno) : ", count)

        strategy = random.choice(HPO_SAMPLE_STRATEGIES)
        precision = min(strategy[0], len(last_hpo_terms))
        imprecision = min(strategy[1], len(last_hpo_pool))

        precise_samples = random.sample(last_hpo_terms, precision)
        imprecise_samples = random.sample(last_hpo_pool, imprecision)

        combined = precise_samples + imprecise_samples

        # no noisy for now

        # add the sampled HPO terms to the dictionary
        sampled_hpoIDs_for_variants[row['#Uploaded_variation']] = combined

        if put_in_the_df:
            df_variants.at[i, 'sampled_hpo'] = ','.join([str(hpo) for hpo in combined])

    # save dictionary to a file
    with open(output_pickle_file, 'wb') as f:
        pickle.dump(sampled_hpoIDs_for_variants, f)

    print("Sampled HPO terms for variants are saved to ", output_pickle_file)
    return sampled_hpoIDs_for_variants, df_variants


def test_sample_hpo_terms_for_variants_optimized():

    df_variants = read_variants()

    print("len of df_variants dataframe: ", len(df_variants))

    gene_dict, hpo_dict = read_dicts()

    # pass the first 100 rows to the function
    sampled_hpoIDs_for_variants, df100 = sample_hpo_terms_for_variants_optimized(
        df_variants.iloc[0:25000], gene_dict, put_in_the_df=True)

    # print the rows of the dataframe (changed)
    for i, row in df100.iterrows():
        print(row['#Uploaded_variation'], row['SYMBOL'], row['sampled_hpo'])

    print(sampled_hpoIDs_for_variants)


def print_num_unique_values_of_categorical_columns(df):
    print("Categorical columns")
    for column in df.select_dtypes(include=['category']).columns:
        print(column, ":", len(df[column].unique()))

    print("Non-categorical columns")
    # for the ones that are not categorical columns
    for column in df.select_dtypes(exclude=['category']).columns:
        print(column, ":", len(df[column].unique()))


def save_categories(df, prefix='categories_', folder_path='../data/categories'):
    # Create the folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    # Select only categorical columns
    categorical_columns = df.select_dtypes(include=['category']).columns
    for column in categorical_columns:
        categories = df[column].cat.categories
        file_path = os.path.join(folder_path, f'{prefix}{column}.csv')
        categories.to_series().to_csv(file_path, index=False)


# Function to apply categories from files to all categorical columns in a new DataFrame
def apply_categories(df, prefix='categories_', folder_path='../data/categories'):
    # Select only categorical columns
    #categorical_columns = df.select_dtypes(include=['category']).columns
    for column in df.columns:
        file_path = os.path.join(folder_path, f'{prefix}{column}.csv')

        if os.path.exists(file_path):
            categories = pd.read_csv(file_path, header=None).squeeze("columns")
            df[column] = pd.Categorical(df[column], categories=categories)


'''
# returns a dictionary with keys as variant names and values as lists of sampled HPO terms
def sample_hpo_terms_for_variants(df_variants, gene_dict, put_in_the_df=False):

    # add new column to the dataframe to store the sampled HPO terms
    if put_in_the_df:
        # np.nan is used to represent missing values
        df_variants['sampled_hpo'] = np.nan

    # get the gene-phenotype relations
    gene_phenotype_relations = get_gene_phenotype_relations()

    # create a network object from hpo_sample
    network = hpo_sample.Network()

    # create dictionary named: sampled_hpoIDs_for_variants
    # key: variant name (e.g. 'rs12345'), value: list of sampled HPO terms
    sampled_hpoIDs_for_variants = {}

    # processed variants
    count = 0
    j = 0
    print("Loop for phenotypes started")
    # for each variant in the dataframe, sample hpo terms using the sample_from_random_strategy function
    for i, row in df_variants.iterrows():
        if j % 1000 == 0:
            print("Row: ", j)
        j += 1

        # get the gene
        gene = row['SYMBOL']

        # if the gene is not in the gene_dict, continue to the next iteration
        if gene not in gene_dict:
            continue

        # get the HPO terms for the gene from the gene-phenotype relations
        hpo_terms = [relation[1] for relation in gene_phenotype_relations if relation[0] == gene]

        if len(hpo_terms) == 0:
            continue

        count += 1
        if count % 10 == 0:
            print("Processed variants (sample pheno) : ", count)
        # sample HPO terms for the gene
        sampled_hpo = network.sample_from_random_strategy(hpo_terms, HPO_SAMPLE_STRATEGIES)

        # add the sampled HPO terms to the dictionary
        sampled_hpoIDs_for_variants[row['#Uploaded_variation']] = sampled_hpo

        if put_in_the_df:
            # add the sampled HPO terms to the dataframe as a new column
            # make it a string to store in the dataframe seperated by comas (values are integers)
            df_variants.at[i, 'sampled_hpo'] = ','.join([str(hpo) for hpo in sampled_hpo])

    return sampled_hpoIDs_for_variants


def test_sample_hpo_terms_for_variants():
    df_variants = read_variants()

    gene_dict, hpo_dict = read_dicts()

    # pass the 22000-22500 rows to the function
    sampled_hpoIDs_for_variants = sample_hpo_terms_for_variants(df_variants.iloc[22000:22500], gene_dict, put_in_the_df=True)
    # print the rows of the dataframe (changed)
    for i, row in df_variants.iloc[22000:22500].iterrows():
        print(row['#Uploaded_variation'], row['SYMBOL'], row['sampled_hpo'])

    print(sampled_hpoIDs_for_variants)'''


def run_binary_classification():
    print("Startanzi")

    # read clean data
    # mostly clean
    df_clean = pd.read_pickle('dataframe.pkl')

    #ValueError: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, The experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:Allele: category, Gene: object, Feature: category, Consequence: category, Existing_variation: object, SYMBOL: category, CANONICAL: category, SIFT: category, PolyPhen: category, HGVSc: object, HGVSp: object, AlphaMissense_score: object, AlphaMissense_pred: object, HGVSc2: object, HGVSp2: object, DS_AG: object, DS_AL: object, DS_DG: object, DS_DL: object, DP_AG: object, DP_AL: object, DP_DG: object, DP_DL: object
    # drop :Gene, Existing_variation: object, SYMBOL: category, CANONICAL: category, SIFT: category, PolyPhen: category, HGVSc: object, HGVSp: object, AlphaMissense_score: object, AlphaMissense_pred: object, HGVSc2: object, HGVSp2: object,
    df_clean = df_clean.drop(columns=['Gene', 'Existing_variation', 'SYMBOL', 'CANONICAL', 'SIFT', 'PolyPhen', 'HGVSc', 'HGVSp', 'AlphaMissense_score', 'AlphaMissense_pred', 'HGVSc2', 'HGVSp2'])
    # make float or nan :  DS_AG: object, DS_AL: object, DS_DG: object, DS_DL: object, DP_AG: object, DP_AL: object, DP_DG: object, DP_DL: object

    # drop uploaded_variation
    df_clean = df_clean.drop(columns=['#Uploaded_variation'])
    df_clean = df_clean.drop(columns=['Feature'])

    # convert the columns to float (pay attention to "None" values)
    # there are some None values in the columns, so we need to replace them with np.nan
    df_clean[['DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL']] = df_clean[['DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL']].replace('None', np.nan).astype('float')

    # print data types and classes of the columns
    print(df_clean.dtypes)
    print(df_clean.select_dtypes(include=['category']).columns)
    print(df_clean.select_dtypes(include=['object']).columns)

    # separate the data into features and target (label is CLIN_SIG)
    y = df_clean['CLIN_SIG']

    # print class distribution
    print(y.value_counts())
    y = y.cat.codes

    X = df_clean.drop(columns=['CLIN_SIG'])

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # imputer: replace missing values with the mean of the column (only for numerical columns)
    # don't use it for categorical columns

    imputer = SimpleImputer(strategy='mean')
    # get the numerical columns
    numerical_columns = X_train.select_dtypes(include=[np.number]).columns
    # fit the imputer to the training data
    imputer.fit(X_train[numerical_columns])

    # The experimental DMatrix parameter`enable_categorical` must be set to `True` to enable categorical split.
    # give the imputed data to the model Xgbboost decision tree
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)


    # print columns of the dataframe
    print(X_train.columns)

    # Train XGBoost model
    params = {
        #'objective': 'binary:logistic',
        "objective": "multi:softmax",
        'eval_metric': 'logloss'
    }
    num_rounds = 100
    model = xgb.train(params, dtrain, num_rounds)

    # Predict probabilities
    y_pred_proba = model.predict(dtest)

    # round the probabilities to get the predicted class
    y_pred = np.round(y_pred_proba)

    # accuracy
    accuracy = (y_pred == y_test).mean()
    print("Accuracy ::", accuracy)

    print("Sample predicted probabilities and answers:", list(zip(y_pred[:100], y_test[:100])))

    # save the model
    model.save_model('../data/modelinzi.xgb')


def run_multilabel(data_path, output_path):
    print("Startanzi")
    # read clean data
    # mostly clean
    # df_clean = pd.read_pickle('dataframe.pkl')
    df_clean = pd.read_pickle(data_path)

    # make HGVSp categorical
    df_clean['HGVSp'] = df_clean['HGVSp'].astype('category')

    # save all categorical codes by getting all categorical columns
    save_categories(df_clean)

    # ValueError: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, The experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:Allele: category, Gene: object, Feature: category, Consequence: category, Existing_variation: object, SYMBOL: category, CANONICAL: category, SIFT: category, PolyPhen: category, HGVSc: object, HGVSp: object, AlphaMissense_score: object, AlphaMissense_pred: object, HGVSc2: object, HGVSp2: object, DS_AG: object, DS_AL: object, DS_DG: object, DS_DL: object, DP_AG: object, DP_AL: object, DP_DG: object, DP_DL: object
    # drop :Gene, Existing_variation: object, SYMBOL: category, CANONICAL: category, SIFT: category, PolyPhen: category, HGVSc: object, HGVSp: object, AlphaMissense_score: object, AlphaMissense_pred: object, HGVSc2: object, HGVSp2: object,
    df_clean = df_clean.drop(
        columns=['Gene', 'Existing_variation', 'CANONICAL', 'AlphaMissense_score', 'AlphaMissense_pred']
    )
    # make float or nan :  DS_AG: object, DS_AL: object, DS_DG: object, DS_DL: object, DP_AG: object, DP_AL: object, DP_DG: object, DP_DL: object

    # drop uploaded_variation
    df_clean = df_clean.drop(columns=['#Uploaded_variation'])
    #df_clean = df_clean.drop(columns=['Feature'])

    # convert the columns to float (pay attention to "None" values)
    # there are some None values in the columns, so we need to replace them with np.nan
    df_clean[['DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL']] = df_clean[
        ['DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL']].replace('None', np.nan).astype(
        'float')

    # separate the data into features and target (label is CLIN_SIG)
    y = df_clean['CLIN_SIG']
    # print class distribution
    print(y.value_counts())
    print("print codes and classes like class 0: benign etc", y.cat.categories)
    y = y.cat.codes
    print(y.value_counts())

    X = df_clean.drop(columns=['CLIN_SIG'])

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # imputer: replace missing values with the mean of the column (only for numerical columns)
    # don't use it for categorical columns

    imputer = SimpleImputer(strategy='mean')
    # get the numerical columns
    numerical_columns = X_train.select_dtypes(include=[np.number]).columns

    # fit the imputer to the training data
    imputer.fit(X_train[numerical_columns])

    # The experimental DMatrix parameter`enable_categorical` must be set to `True` to enable categorical split.
    # give the imputed data to the model Xgbboost decision tree
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    # print columns of dtrain
    print(dtrain.feature_names)

    # print columns of the dataframe
    print(X_train.columns)

    # Train XGBoost model
    """params = {
        "objective": "multi:softmax",
        "eval_metric": 'mlogloss',  # 'logloss' is for binary classification. Use 'mlogloss' for multi-class.
        "num_class": 4  # Assuming there are 4 classes. Adjust according to your dataset.
    }
    num_rounds = 500
    model = xgb.train(params, dtrain, num_rounds)"""

    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": 4,  # Assuming there are 4 classes. Adjust according to your dataset.
        #"max_depth": 8,  # Control the depth of the tree to prevent overfitting
        #"min_child_weight": 1,  # Minimum sum of instance weight (hessian) needed in a child
        #"gamma": 0.1,  # Minimum loss reduction required to make a further partition on a leaf node
        #"subsample": 0.8,  # Subsample ratio of the training instances.
        #"colsample_bytree": 0.8,  # Subsample ratio of columns when constructing each tree.
        #"lambda": 1,  # L2 regularization term on weights
        #"alpha": 0  # L1 regularization term on weights
    }

    num_rounds = 1000
    early_stopping_rounds = 50  # Stop if no improvement after this many rounds

    # Training with early stopping
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=[(dtrain, 'train'), (dtest, 'eval')],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=True
    )

    print(f"Best iteration: {model.best_iteration + 1}")

    # save the model best iteration
    model.save_model(output_path)


def add_embedding_info(
    df_variants,
    path_to_embedding,
    path_to_gene_dict,
    path_to_hpo_dict,
    path_to_sampled_hpo,
    add_scaled_average_dot_product=True,
    add_scaled_min_dot_product=False,
    add_scaled_max_dot_product=False,
    add_average_dot_product=True,
    add_min_dot_product=False,
    add_max_dot_product=False,
    add_std_dot_product=False,
    add_gene_embedding=False,
    add_fix_num_phen_embedding=0
):

    # read the embeddings
    node_embeddings = read_embedding(path_to_embedding)
    # read the gene and hpo dictionaries
    gene_dict, hpo_dict = read_dicts(path_to_gene_dict, path_to_hpo_dict)

    # read sampled hpo
    with open(path_to_sampled_hpo, 'rb') as f:
        sampled_hpoIDs_for_variants = pickle.load(f)

    if add_fix_num_phen_embedding > 0:
        global neutral_embedding
        # calculate the neutral embedding as the root embedding
        neutral_embedding = calculate_neutral_embedding_as_root_embedding(node_embeddings, hpo_dict)

    if add_scaled_average_dot_product:
        df_variants['scaled_average_dot_product'] = np.nan
    if add_scaled_min_dot_product:
        df_variants['scaled_min_dot_product'] = np.nan
    if add_scaled_max_dot_product:
        df_variants['scaled_max_dot_product'] = np.nan
    if add_average_dot_product:
        df_variants['average_dot_product'] = np.nan
    if add_min_dot_product:
        df_variants['min_dot_product'] = np.nan
    if add_max_dot_product:
        df_variants['max_dot_product'] = np.nan
    if add_std_dot_product:
        df_variants['std_dot_product'] = np.nan
    if add_gene_embedding:
        df_variants['gene_embedding'] = np.nan
    if add_fix_num_phen_embedding > 0:
        for i in range(add_fix_num_phen_embedding):
            df_variants[f'phen_embedding_{i}'] = np.nan

    # for every row add necessary columns to the dataframe
    for i, row in df_variants.iterrows():

        gene = row['SYMBOL']

        if gene not in gene_dict:
            continue

        # get the gene embedding
        gene_embedding = node_embeddings[gene_dict[gene]]

        # if variant has no sampled HPO terms, continue to the next iteration
        if row['#Uploaded_variation'] not in sampled_hpoIDs_for_variants:
            continue

        #  get the sampled HPO terms for the variant
        sampled_hpo = sampled_hpoIDs_for_variants[row['#Uploaded_variation']]

        # get the embeddings of the HPO terms
        hpo_embeddings = [node_embeddings[hpo_dict[hpo]] for hpo in sampled_hpo]

        dot_products = [np.dot(gene_embedding, hpo_embedding) for hpo_embedding in hpo_embeddings]

        # scaled dot products of vectors
        scaled_dot_products = [dot_product / (np.linalg.norm(gene_embedding) * np.linalg.norm(hpo_embedding)) for dot_product, hpo_embedding in zip(dot_products, hpo_embeddings)]

        if add_scaled_average_dot_product:
            df_variants.at[i, 'scaled_average_dot_product'] = np.mean(scaled_dot_products)

        if add_scaled_min_dot_product:
            df_variants.at[i, 'scaled_min_dot_product'] = np.min(scaled_dot_products)

        if add_scaled_max_dot_product:
            df_variants.at[i, 'scaled_max_dot_product'] = np.max(scaled_dot_products)

        # if asked, add average dot product between gene and hpo embeddings
        if add_average_dot_product:
            df_variants.at[i, 'average_dot_product'] = np.mean(dot_products)

        # if asked, add min dot product between gene and hpo embeddings
        if add_min_dot_product:
            df_variants.at[i, 'min_dot_product'] = np.min(dot_products)

        # if asked, add max dot product between gene and hpo embeddings
        if add_max_dot_product:
            df_variants.at[i, 'max_dot_product'] = np.max(dot_products)

        # if asked, add std dot product between gene and hpo embeddings
        if add_std_dot_product:
            df_variants.at[i, 'std_dot_product'] = np.std(dot_products)

        if add_gene_embedding:
            df_variants.at[i, 'gene_embedding'] = gene_embedding

        if add_fix_num_phen_embedding > 0:
            # add the neutral embedding to the list until the list has fix_num_phen_embedding elements
            hpo_embeddings = hpo_embeddings + [neutral_embedding] * (add_fix_num_phen_embedding - len(hpo_embeddings))
            # add the embeddings to the dataframe
            for j in range(add_fix_num_phen_embedding):
                df_variants.at[i, f'phen_embedding_{j}'] = hpo_embeddings[j]

    return df_variants


# read the cleaned datadf_variants = pd.read_pickle('../data/variants_cleaned.pkl')

# read dictionaries
#gene_dict, hpo_dict = read_dicts()

# add the sampled HPO terms to the dataframe (frequency based)
#sample_hpo_terms_with_frequency_optimized(df_variants, gene_dict, '../data/sampled_hpoIDs__with_freq_for_variants.pkl', put_in_the_df=False)


# run multilabel classification



#run_multilabel('../data/df_with_first_embedding_freq_based.pkl', '../data/model_with_first_embedding_freq_based.xgb')
#exit()



#df = pd.read_pickle('../data/df_with_first_embedding_freq_based.pkl')

"""print("len of df: ", len(df))

print_num_unique_values_of_categorical_columns(df)

# print some values of the dataframe from every column
for column in df.select_dtypes(include=['category']).columns:
    print(column, ":", df[column].unique()[:10])



exit()"""


def add_embo(
    path_to_sampled_hpo='../data/sampled_hpoIDs_with_freq_for_variants.pkl'
):
    print("Startanzi")
    df_variants = pd.read_pickle('../data/variants_cleaned.pkl')

    # add embedding info to the dataframe
    df_variants = add_embedding_info(
        df_variants,
        path_to_embedding='../data/node_embeddings.txt',
        path_to_gene_dict='../data/gene_dict.pt',
        path_to_hpo_dict='../data/hpo_dict.pt',
        path_to_sampled_hpo=path_to_sampled_hpo,
        add_scaled_average_dot_product=True,
        add_scaled_min_dot_product=True,
        add_scaled_max_dot_product=True,
        add_average_dot_product=True,
        add_min_dot_product=True,
        add_max_dot_product=True
    )

    print(df_variants.head(10))

    # eliminate the rows that has none as average_dot_product
    df_variants = df_variants.dropna(subset=['average_dot_product'])
    print(df_variants.head(10))

    # save df_variants to a new file in data folder outside the repository
    df_variants.to_pickle('../data/df_with_first_embedding_freq_based.pkl')
    print("Dataframe with embedding is saved to dataframe_with_embedding.pkl")