import random
import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import torch
import helpers.hpo_sample
from helpers.gene_mapping import get_gene_phenotype_relations

path_variants = '../data/5bc6f943-66e0-4254-94f5-ed3888f05d0a.vep.tsv'

path_embedding = '../data/node_embeddings.txt'

path_gene_dict = '../data/gene_dict.pt'
path_hpo_dict = '../data/hpo_dict.pt'

neutral_embedding = None

def read_variants():
    # read by skippin 48 lines
    df = pd.read_csv(path_variants, sep='\t', skiprows=48)
    return df


def read_embedding(path_of_embedding = path_embedding):
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



def read_dicts(gene_path = path_gene_dict, hpo_path = path_hpo_dict):
    gene_dict = torch.load(path_gene_dict)
    hpo_dict = torch.load(path_hpo_dict)
    return gene_dict, hpo_dict

#symbols = df['SYMBOL']

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

# get phenetype gene relations


def add_sampled_hpo_terms_full_precise(df_variants, node_embeddings ,gene_dict, hpo_dict, number_of_phenotypes):
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
def get_model(df_training):
    X = [3]
    y = [4] # sill
    # train the xgboost model
    # model
    # Imputation
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    num_rounds = 100
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    model = xgb.train(params, dtrain, num_rounds)

    # Predict probabilities
    y_pred_proba = model.predict(dtest)

    print("Sample predicted probabilities:", y_pred_proba[:10])


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
def clean_data(df_variants):
    # only get the rows with pathogenic or benign, likely_benign, likely_pathogenic values in CLIN_SIG column
    df_variants = df_variants[df_variants['CLIN_SIG'].isin(['pathogenic', 'benign', 'likely_benign', 'likely_pathogenic'])]
    # divide  Hgvsc  columns into two columns by splitting the values by ':'
    df_variants[['HGVSc', 'HGVSc2']] = df_variants['HGVSc'].str.split(':', expand=True)
    # divide  Hgvsp  columns into two columns by splitting the values by ':'
    df_variants[['HGVSp', 'HGVSp2']] = df_variants['HGVSp'].str.split(':', expand=True)

    # divide  SpliceAI_pred  columns into 8 columns by splitting the values by '|' and drop the first column
    df_variants[['SpliceAI_pred_symbol', 'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL']] = df_variants['SpliceAI_pred'].str.split('|', expand=True)
    df_variants = df_variants.drop(columns=['SpliceAI_pred', 'SpliceAI_pred_symbol'])

    # divide alpha missense columns has several (different) values seperated by commas, replace the values as mean of the values and standard deviation of the values (pay attention to "-" values)
    # note that number of values are not the same for each row, so we cannot use str.
    """
    df[['mean', 'std_dev']] = df['values'].apply(
        lambda x: (np.nan, np.nan) if x == "-" else 
        ((mean := np.mean(numbers := np.array(x.split(',')).astype(float))), np.std(numbers))
    ).apply(pd.Series)
    """

    """
    def find_problematic_rows(df):
        problematic_rows = []
        for index, row in df.iterrows():
            try:
                # Attempt to convert each value in the 'values' column to float, after splitting by comma
                [float(num) for num in row['AlphaMissense_score'].split(',') if num != "-"]
            except ValueError:
                # If a ValueError occurs, add the row index to the problematic_rows list
                problematic_rows.append(index)
        return problematic_rows
    
    # Find and print problematic rows
    problematic_rows = find_problematic_rows(df_variants)
    if problematic_rows:
        print("Problematic rows found at indices:", problematic_rows)
        print("Problematic data:", df_variants.loc[problematic_rows])
    else:
        print("No problematic rows found.")"""



    # alpha missense scorelarda . lar var

    # eliminate ,., values for example 0.0848,.,0.0854 -> 0.0848,0.0854 or .,0.5 -> 0.5  or 0.5,. -> 0.5 so, repleca . , pairs that is adjacent to each other




    df_variants[['AlphaMissense_score_mean', 'AlphaMissense_std_dev']] = df_variants['AlphaMissense_score'].apply(
        lambda x: (np.nan, np.nan) if (x == "-" or x == "") else
        # Filter the split results to include only valid floats, then perform calculations
        ((numbers := np.array([float(num) for num in x.split(',') if num.replace('.', '', 1).isdigit()]),
          np.mean(numbers), np.std(numbers))[1:]
         if any(num.replace('.', '', 1).isdigit() for num in x.split(','))
         else (np.nan, np.nan))
    ).apply(pd.Series)



    # alpha missense pred has several values ("A", "B", "P") seperated by commas, change this column to ratio of letters (pay attention to "-" values)
    df_variants['AlphaMissense_pred'] = df_variants['AlphaMissense_pred'].apply(lambda x: (x.count('A'), x.count('B'), x.count('P')) if x != "-" else (np.nan, np.nan, np.nan))
    # divide it so that we have 3 columns for each letter storing the ratio of the letter in the column if 3 A's 2 B's and 1 P's then 3/6, 2/6, 1/6
    df_variants[['AlphaMissense_pred_A', 'AlphaMissense_pred_B', 'AlphaMissense_pred_P']] = df_variants['AlphaMissense_pred'].apply(
        lambda x: (np.nan, np.nan, np.nan) if x == "-" or sum(x) == 0 else (x[0] / sum(x), x[1] / sum(x), x[2] / sum(x))).apply(pd.Series)




    # for each column print the different values in the first 100 rows and number of different values in the column (as a whole)
    for column in df_variants.columns:
        print("--------------------")
        print(column)
        for value in df_variants[column].head(100):
            print(value)
        print(df_variants[column].nunique())

    # make the column canonical a categorical column for model
    df_variants['CANONICAL'] = df_variants['CANONICAL'].astype('category')

    # make the Allele column a categorical column for model
    df_variants['Allele'] = df_variants['Allele'].astype('category')

    # make the column polyphen a numerical column and categorical for model by extracting the number from the string and taking the string before by looking for the '(number)'
    df_variants['PolyPhen_number'] = df_variants['PolyPhen'].str.extract(r'\((\d+\.\d+)\)').astype('float')
    df_variants['PolyPhen'] = df_variants['PolyPhen'].str.extract(r'(\w+)').astype('category')

    # make the column SIFT a numerical column and categorical for model by extracting the number from the string and taking the string before by looking for the '(number)'
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

    # check the columns again
    for column in df_variants.columns:
        print("-NEWWWWW-------------------")
        print(column)
        for value in df_variants[column].head(100):
            print(value)
        print(df_variants[column].nunique())

    # save the dataframe to a new file that preseves data types (category, numerical etc for each column)
    # pickle the dataframe
    df_variants.to_pickle('multilabel_clean.pkl')


# call
#df_variants = read_variants()
#clean_data(df_variants)
#exit(0)

def run_binary_classification():
    print("Startanzi")
    # read clean data
    # mostly clean
    df_clean = pd.read_pickle('dataframe.pkl')
    #df_clean = pd.read_pickle('multilabel_clean.pkl')

    # sample 5 HPO terms for each gene and add the embeddings to the dataframe
    # get the embeddings of the HPO terms
    #node_embeddings = read_embedding()
    # read the gene and hpo dictionaries
    #gene_dict, hpo_dict = read_dicts()
    # calculate the neutral embedding as the root embedding
    #neutral_embedding = calculate_neutral_embedding_as_root_embedding(node_embeddings, hpo_dict)







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
    #print class distribution
    print(y.value_counts())
    y = y.cat.codes

    X = df_clean.drop(columns=['CLIN_SIG'])


    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # imputer: replace missing values with the mean of the column (only for numerical columns)
    # dont use it for categorical columns

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


def run_multilabel():
    print("Startanzi")
    # read clean data
    # mostly clean
    # df_clean = pd.read_pickle('dataframe.pkl')
    df_clean = pd.read_pickle('multilabel_clean.pkl')

    # sample 5 HPO terms for each gene and add the embeddings to the dataframe
    # get the embeddings of the HPO terms
    # node_embeddings = read_embedding()
    # read the gene and hpo dictionaries
    # gene_dict, hpo_dict = read_dicts()
    # calculate the neutral embedding as the root embedding
    # neutral_embedding = calculate_neutral_embedding_as_root_embedding(node_embeddings, hpo_dict)

    # ValueError: DataFrame.dtypes for data must be int, float, bool or category. When categorical type is supplied, The experimental DMatrix parameter`enable_categorical` must be set to `True`.  Invalid columns:Allele: category, Gene: object, Feature: category, Consequence: category, Existing_variation: object, SYMBOL: category, CANONICAL: category, SIFT: category, PolyPhen: category, HGVSc: object, HGVSp: object, AlphaMissense_score: object, AlphaMissense_pred: object, HGVSc2: object, HGVSp2: object, DS_AG: object, DS_AL: object, DS_DG: object, DS_DL: object, DP_AG: object, DP_AL: object, DP_DG: object, DP_DL: object
    # drop :Gene, Existing_variation: object, SYMBOL: category, CANONICAL: category, SIFT: category, PolyPhen: category, HGVSc: object, HGVSp: object, AlphaMissense_score: object, AlphaMissense_pred: object, HGVSc2: object, HGVSp2: object,
    df_clean = df_clean.drop(
        columns=['Gene', 'Existing_variation', 'SYMBOL', 'CANONICAL', 'SIFT', 'PolyPhen', 'HGVSc', 'HGVSp',
                 'AlphaMissense_score', 'AlphaMissense_pred', 'HGVSc2', 'HGVSp2'])
    # make float or nan :  DS_AG: object, DS_AL: object, DS_DG: object, DS_DL: object, DP_AG: object, DP_AL: object, DP_DG: object, DP_DL: object

    # drop uploaded_variation
    df_clean = df_clean.drop(columns=['#Uploaded_variation'])
    df_clean = df_clean.drop(columns=['Feature'])

    # convert the columns to float (pay attention to "None" values)
    # there are some None values in the columns, so we need to replace them with np.nan
    df_clean[['DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL']] = df_clean[
        ['DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL']].replace('None', np.nan).astype(
        'float')

    # print data types and classes of the columns
    print(df_clean.dtypes)
    print(df_clean.select_dtypes(include=['category']).columns)
    print(df_clean.select_dtypes(include=['object']).columns)

    # separate the data into features and target (label is CLIN_SIG)
    y = df_clean['CLIN_SIG']
    # print class distribution
    print(y.value_counts())
    print("print codes and classes like class 0: benign etc", y.cat.categories)
    y = y.cat.codes
    print(y.value_counts())
    #exit()

    X = df_clean.drop(columns=['CLIN_SIG'])

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # imputer: replace missing values with the mean of the column (only for numerical columns)
    # dont use it for categorical columns

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
        "objective": "multi:softmax",
        "eval_metric": 'mlogloss',  # 'logloss' is for binary classification. Use 'mlogloss' for multi-class.
        "num_class": 4  # Assuming there are 4 classes. Adjust according to your dataset.
    }
    num_rounds = 500
    model = xgb.train(params, dtrain, num_rounds)

    # Predict probabilities
    y_pred_proba = model.predict(dtest)

    # round the probabilities to get the predicted class
    y_pred = np.round(y_pred_proba)

    # accuracy
    accuracy = (y_pred == y_test).mean()
    print("Accuracy ::", accuracy)

    # print accuracy of each class
    print("Accuracy of each class:", (y_pred == y_test).groupby(y_test).mean())

    # print precision, recall, and F1 score
    print(classification_report(y_test, y_pred))

    print("Sample predicted probabilities and answers:", list(zip(y_pred[:100], y_test[:100])))

    # save the model
    model.save_model('../data/multilabal_modelinzi.xgb')





#run_binary_classification()
run_multilabel()