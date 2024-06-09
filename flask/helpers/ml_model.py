import pickle
import random
import xgboost as xgb
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import torch
from helpers.gene_mapping import get_gene_phenotype_relations, get_gene_phenotype_relations_and_frequency
from helpers.xgboost_train import apply_categories, read_embedding, read_dicts, read_variants

model_path = 'data/model_with_first_embedding_freq_based.xgb'

embedding_path = 'data/node_embeddings.txt'

path_gene_dict = 'data/gene_dict.pt'
path_hpo_dict = 'data/hpo_dict.pt'

# calculate the relative paths


# Load the gene dictionary using train1
gene_dictionary = torch.load(path_gene_dict)
hpo_dictionary = torch.load(path_hpo_dict)

# Load the embeddings using train1
embeddings = read_embedding(embedding_path)


training_data_columns = ['Allele', 'Feature', 'Consequence', 'SYMBOL', 'SIFT', 'PolyPhen',
       'HGVSp', 'AF', 'gnomADe_AF', 'REVEL', 'DANN_score', 'MetaLR_score',
       'CADD_raw_rankscore', 'ExAC_AF', 'ALFA_Total_AF',
       'turkishvariome_TV_AF', 'HGVSc_number', 'HGVSc_change', 'HGVSp_number',
       'HGVSp_change', 'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL',
       'DP_DG', 'DP_DL', 'AlphaMissense_score_mean', 'AlphaMissense_std_dev',
       'AlphaMissense_pred_A', 'AlphaMissense_pred_B', 'AlphaMissense_pred_P',
       'PolyPhen_number', 'SIFT_number', 'scaled_average_dot_product',
       'scaled_min_dot_product', 'scaled_max_dot_product',
       'average_dot_product', 'min_dot_product', 'max_dot_product']

# parameters for add embedding info to patient
parameters = {
    "average_dot_product": True,
    "min_dot_product": False,
    "max_dot_product": False,
    "std_dot_product": False,
    "scaled_average_dot_product": True,
    "scaled_min_dot_product": False,
    "scaled_max_dot_product": False,
    "gene_embedding": False,
    "fix_num_phen_embedding": 0
}

# check training data columns to set parameters
for param in parameters:

    if param == "fix_num_phen_embedding":
        continue

    if param in training_data_columns:
        parameters[param] = True
    else:
        parameters[param] = False


# Load the xgboost model
model = xgb.Booster()
model.load_model(model_path)


def prepare_data(df_variants):

    # HGSVc_original column is the same as HGVSc column
    df_variants['HGSVc_original'] = df_variants['HGVSc']

    # HGSVp_original column is the same as HGVSp column
    df_variants['HGSVp_original'] = df_variants['HGVSp']

    # Conseqence_original column is the same as Consequence column
    df_variants['Consequence_original'] = df_variants['Consequence']

    # turkishvariome_TV_AF_original column is the same as turkishvariome_TV_AF column
    df_variants['turkishvariome_TV_AF_original'] = df_variants['turkishvariome_TV_AF']

    # Gene_original column is the same as Gene column
    df_variants['Gene_original'] = df_variants['Gene']

    # Allele original column is the same as Allele column
    df_variants['Allele_original'] = df_variants['Allele']

    # divide  HGVSc  columns into two columns by splitting the values by ':' 	ENST00000616125.5:c.11G>A
    df_variants[['HGVSc', 'HGVSc2']] = df_variants['HGVSc'].str.split(':', expand=True)

    # extract the info from the HGVSc2 column so that we have the 2 more columns 11, G>A # ENST00000616125.5:c.11G>A
    df_variants[['HGVSc_number', 'HGVSc_change']] = df_variants['HGVSc2'].str.extract(r'c\.(\d+)([A-Z]>.*)')

    # divide  HGVSp  columns into two columns by splitting the values by ':' 	ENSP00000484643.1:p.Gly4Glu
    try:
        df_variants[['HGVSp', 'HGVSp2']] = df_variants['HGVSp'].str.split(':', expand=True)
    except:
        # put the values as None if there is an error
        df_variants['HGVSp'] = None
        df_variants['HGVSp2'] = None
        print("Unusual HGVSp column")

    # extract the info from the HGVSp2 column so that we have the 2 more columns 4, GlyGlu namely number in the middle and the change
    # pay attention that there should be 2 columns
    df_variants[['HGVSp_from', 'HGVSp_number', 'HGVSp_to']] = df_variants['HGVSp2'].str.extract(
        r'p\.([A-Z][a-z]+)(\d+)([A-Z][a-z]+)')

    # combine from and to columns to get the change
    df_variants['HGVSp_change'] = df_variants['HGVSp_from'] + df_variants['HGVSp_to']

    # drop the columns that are not needed
    df_variants = df_variants.drop(columns=['HGVSc', 'HGVSc2', 'HGVSp2', 'HGVSp_from', 'HGVSp_to'])

    # divide  SpliceAI_pred  columns into 8 columns by splitting the values by '|' and drop the first column
    df_variants[['SpliceAI_pred_symbol', 'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL']] = \
    df_variants['SpliceAI_pred'].str.split('|', expand=True)
    #df_variants = df_variants.drop(columns=['SpliceAI_pred', 'SpliceAI_pred_symbol'])

    df_variants[['AlphaMissense_score_mean', 'AlphaMissense_std_dev']] = df_variants['AlphaMissense_score'].apply(
        lambda x: (np.nan, np.nan) if (x == "-" or x == "") else
        # Filter the split results to include only valid floats, then perform calculations
        ((numbers := np.array([float(num) for num in x.split(',') if num.replace('.', '', 1).isdigit()]),
          np.mean(numbers), np.std(numbers))[1:]
         if any(num.replace('.', '', 1).isdigit() for num in x.split(','))
         else (np.nan, np.nan))
    ).apply(pd.Series)

    # alpha missense pred has several values ("A", "B", "P") seperated by commas, change this column to ratio of letters (pay attention to "-" values)
    df_variants['AlphaMissense_pred'] = df_variants['AlphaMissense_pred'].apply(
        lambda x: (x.count('A'), x.count('B'), x.count('P')) if x != "-" else (np.nan, np.nan, np.nan))

    # divide it so that we have 3 columns for each letter storing the ratio of the letter in the column if 3 A's 2 B's and 1 P's then 3/6, 2/6, 1/6
    df_variants[['AlphaMissense_pred_A', 'AlphaMissense_pred_B', 'AlphaMissense_pred_P']] = df_variants[
        'AlphaMissense_pred'].apply(
        lambda x: (np.nan, np.nan, np.nan) if x == "-" or sum(x) == 0 else (
        x[0] / sum(x), x[1] / sum(x), x[2] / sum(x))).apply(pd.Series)

    # for categorical columns, we need to convert them to numerical columns which should overlap with the training data
    apply_categories(df_variants, folder_path="data/categories")
    print(df_variants.dtypes)

    # HGSV.. columns
    # make the column HGVSc_number a numerical column for model (pay attention to "-" values)
    df_variants['HGVSc_number'] = df_variants['HGVSc_number'].replace('-', np.nan).astype('float')

    # make the column HGVSp_number a numerical column for model (pay attention to "-" values)
    df_variants['HGVSp_number'] = df_variants['HGVSp_number'].replace('-', np.nan).astype('float')

    # make the column HGVSp_change a categorical column for model
    #df_variants['HGVSp_change'] = df_variants['HGVSp_change'].astype('category')

    # make the column HGVSc_change a categorical column for model
    #df_variants['HGVSc_change'] = df_variants['HGVSc_change'].astype('category')

    # new here
    #df_variants['HGVSp'] = df_variants['HGVSp'].astype('category')

    # make the column canonical a categorical column for model
    #df_variants['CANONICAL'] = df_variants['CANONICAL'].astype('category')

    # make the Allele column a categorical column for model
    #df_variants['Allele'] = df_variants['Allele'].astype('category')

    # make the column polyphen a numerical column and categorical for model by extracting the number from the string and taking the string before by looking for the '(number)'
    df_variants['PolyPhen_number'] = df_variants['PolyPhen'].str.extract(r'\((\d+\.\d+)\)').astype('float')
    #df_variants['PolyPhen'] = df_variants['PolyPhen'].str.extract(r'(\w+)').astype('category')

    # make the column SIFT a numerical column and categorical for model by extracting the number from the string and taking the string before by looking for the '(number)'
    df_variants['SIFT_number'] = df_variants['SIFT'].str.extract(r'\((\d+\.\d+)\)').astype('float')
    #df_variants['SIFT'] = df_variants['SIFT'].str.extract(r'(\w+)').astype('category')

    # make the gene column a categorical column for model
    #df_variants['SYMBOL'] = df_variants['SYMBOL'].astype('category')

    # make the column Feature a categorical column for model
    #df_variants['Feature'] = df_variants['Feature'].astype('category')

    # make the column Consequence a categorical column for model
    #df_variants['Consequence'] = df_variants['Consequence'].astype('category')

    # make the column Existing_variation (values are seperated by commas) a categorical column for model by splitting the values
    # df_variants['Existing_variation'] = df_variants['Existing_variation'].str.split(',').explode().astype('category')

    # make the column AF a numerical column for model (pay attention to "-" values)
    df_variants['AF'] = df_variants['AF'].replace('-', np.nan).astype('float')

    # make the column gnomADe_AF a numerical column for model (pay attention to "-" values)
    df_variants['gnomADe_AF'] = df_variants['gnomADe_AF'].replace('-', np.nan).astype('float')

    # make the column CLIN_SIG a categorical column for model
    #df_variants['CLIN_SIG'] = df_variants['CLIN_SIG'].astype('category')

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

    #df_clean = df_clean.drop(columns=['Gene', 'Existing_variation', 'SYMBOL', 'CANONICAL', 'SIFT', 'PolyPhen', 'HGVSc', 'HGVSp','AlphaMissense_score', 'AlphaMissense_pred', 'HGVSc2', 'HGVSp2'])
    # make float or nan :  DS_AG: object, DS_AL: object, DS_DG: object, DS_DL: object, DP_AG: object, DP_AL: object, DP_DG: object, DP_DL: object

    # drop uploaded_variation
    #df_clean = df_clean.drop(columns=['#Uploaded_variation'])
    #df_clean = df_clean.drop(columns=['Feature'])

    # convert the columns to float (pay attention to "None" values)
    # there are some None values in the columns, so we need to replace them with np.nan
    df_variants[['DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL']] = df_variants[
        ['DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL']].replace('None', np.nan).astype(
        'float')

    imputer = SimpleImputer(strategy='mean')
    # get the numerical columns
    numerical_columns = df_variants.select_dtypes(include=[np.number]).columns
    # fit the imputer
    imputer.fit(df_variants[numerical_columns])

    # transform the data  !!!!!!!!!!
    #df_variants[numerical_columns] = imputer.transform(df_variants[numerical_columns])

    return df_variants


def add_embedding_info_to_patient(gene, sampled_hpo,
                       add_scaled_average_dot_product=True, add_scaled_min_dot_product=False, add_scaled_max_dot_product=False,
                       add_average_dot_product = True, add_min_dot_product = False, add_max_dot_product = False, add_std_dot_product = False, add_gene_embedding = False, add_fix_num_phen_embedding = 0):

    global gene_dictionary, hpo_dictionary, embeddings

    if add_fix_num_phen_embedding > 0:
        pass

    hpo_embedding_info = {}

    # get the gene embedding
    if gene not in gene_dictionary:
        return hpo_embedding_info

    gene_embedding = embeddings[gene_dictionary[gene]]

    # get the embeddings of the HPO terms
    hpo_embeddings = [embeddings[hpo_dictionary[hpo]] for hpo in sampled_hpo]

    dot_products = [np.dot(gene_embedding, hpo_embedding) for hpo_embedding in hpo_embeddings]

    # scaled dot products of vectors
    scaled_dot_products = [dot_product / (np.linalg.norm(gene_embedding) * np.linalg.norm(hpo_embedding)) for dot_product, hpo_embedding in zip(dot_products, hpo_embeddings)]

    if add_scaled_average_dot_product:
        hpo_embedding_info['scaled_average_dot_product'] = np.mean(scaled_dot_products)

    if add_scaled_min_dot_product:
        hpo_embedding_info['scaled_min_dot_product'] = np.min(scaled_dot_products)
    if add_scaled_max_dot_product:
        hpo_embedding_info['scaled_max_dot_product'] = np.max(scaled_dot_products)

    # if asked, add average dot product between gene and hpo embeddings
    if add_average_dot_product:
        hpo_embedding_info['average_dot_product'] = np.mean(dot_products)

    # if asked, add min dot product between gene and hpo embeddings
    if add_min_dot_product:
        hpo_embedding_info['min_dot_product'] = np.min(dot_products)
    # if asked, add max dot product between gene and hpo embeddings
    if add_max_dot_product:
        hpo_embedding_info['max_dot_product'] = np.max(dot_products)
    # if asked, add std dot product between gene and hpo embeddings
    if add_std_dot_product:
        hpo_embedding_info['std_dot_product'] = np.std(dot_products)
    if add_gene_embedding:
        hpo_embedding_info['gene_embedding'] = gene_embedding
    if add_fix_num_phen_embedding > 0:
        pass

    return hpo_embedding_info


# function takes  the patient's variants and runs the model on them and returns the ranking of the variants
def add_model_scores(variants, hpo_term_ids):

    # prepare the data
    variants = prepare_data(variants)

    # for every row in the variants, insert the hpo related columns
    for index, row in variants.iterrows():
        # get necessary params from parameters
        hpo_embedding_info = add_embedding_info_to_patient(
            row['SYMBOL'],
            hpo_term_ids,
            add_scaled_average_dot_product=parameters["scaled_average_dot_product"],
            add_scaled_min_dot_product=parameters["scaled_min_dot_product"],
            add_scaled_max_dot_product=parameters["scaled_max_dot_product"],
            add_average_dot_product=parameters["average_dot_product"],
            add_min_dot_product=parameters["min_dot_product"],
            add_max_dot_product=parameters["max_dot_product"],
            add_std_dot_product=parameters["std_dot_product"],
            add_gene_embedding=parameters["gene_embedding"],
            add_fix_num_phen_embedding=parameters["fix_num_phen_embedding"]
        )

        # add the columns to dataframe if they are not already in the dataframe
        for key in hpo_embedding_info.keys():
            if key not in variants.columns:
                variants[key] = np.nan

        # add the embedding info in hpo_embedding_info to the row
        for key, value in hpo_embedding_info.items():
            variants.at[index, key] = value

    # insert hpo related columns
    #hpo_embedding_info = add_embedding_info_to_patient(variants['SYMBOL'][0], hpo_term_ids, add_scaled_average_dot_product=True, add_scaled_min_dot_product=False, add_scaled_max_dot_product=False,
                        #add_average_dot_product = True, add_min_dot_product = False, add_max_dot_product = False, add_std_dot_product = False, add_gene_embedding = False, add_fix_num_phen_embedding = 0)

    # add the embedding info in hpo_embedding_info to the variants all rows
    #variants = pd.concat([variants, pd.DataFrame([hpo_embedding_info])], axis=1)

    # get the columns that are needed for the model
    columns = training_data_columns

    # run the model and append the results to the variants
    # convert the variants[columns] to DMatrix object
    data_dmatrix = xgb.DMatrix(data=variants[columns], enable_categorical=True)

    # get the predictions
    scores = model.predict(data_dmatrix)
    # according to  scores = scores[:, 3] + scores[:, 2] * 0.6 - scores[:, 0] - scores[:, 1] * 0.6
    variants['Priovar_score'] = scores[:, 3] + scores[:, 2] * 0.6 - scores[:, 0] - scores[:, 1] * 0.6

    return variants


def test_add_model_scores():
    path_variants = '../data/5bc6f943-66e0-4254-94f5-ed3888f05d0a.vep.tsv'

    # read the variants
    variants = df = pd.read_csv(path_variants, sep='\t', skiprows=48)

    # get random 10 rows
    variants = variants.sample(10)

    example_res = add_model_scores(variants, [1, 2])

    print(example_res)


# mock model results
def get_mock_results(num_variants=3, hpo=None):

    # read the variants
    path_variants = 'data/5bc6f943-66e0-4254-94f5-ed3888f05d0a.vep.tsv'
    variants = df = pd.read_csv(path_variants, sep='\t', skiprows=48)
    variants = variants.sample(num_variants)

    if hpo is None:
        hpo = [4000104, 4000105, 4000106, 4000107, 4000108]
        variants = add_model_scores(variants, hpo)
    else:
        variants = add_model_scores(variants, hpo)

    # scale Priovar_score (-1.6, 1.6) to be between 0 and 1
    variants['Priovar_score'] = (variants['Priovar_score'] + 1.6) / 3.2

    return variants


def get_real_results(path_variants, hpo):

    # read the variants
    variants = pd.read_csv(path_variants, sep='\t', skiprows=48)

    variants = add_model_scores(variants, hpo)

    # scale Priovar_score (-1.6, 1.6) to be between 0 and 1
    variants['Priovar_score'] = (variants['Priovar_score'] + 1.6) / 3.2

    return variants

#test_add_model_scores()




























