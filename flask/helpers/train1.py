import random
import xgboost as xgb
import numpy as np
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


def read_embedding():
    node_embeddings = np.loadtxt(path_embedding, skiprows=1)

    # sort the array by the first column
    node_embeddings = node_embeddings[node_embeddings[:, 0].argsort()]

    # remove the first column
    node_embeddings = node_embeddings[:, 1:]
    return node_embeddings

def calculate_neutral_embedding_as_root_embedding(node_embeddings, hpo_dict):
    # get the embedding of first phenotype
    root_embedding = node_embeddings[hpo_dict[1]]
    return root_embedding



def read_dicts():
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



# read the variants
df_variants = read_variants()

# show class distribution for CLIN_SIG column
dist = df_variants['CLIN_SIG'].value_counts()

# only get the rows with pathogenic or benign values in CLIN_SIG column
df_variants = df_variants[df_variants['CLIN_SIG'].isin(['pathogenic', 'benign'])]
print("size of df_variants: ", df_variants.shape)

# read the node embeddings
node_embeddings = read_embedding()

# read the gene and hpo dictionaries
gene_dict, hpo_dict = read_dicts()

# calculate the neutral embedding as the root embedding
neutral_embedding = calculate_neutral_embedding_as_root_embedding(node_embeddings, hpo_dict)

# get the types of the columns
print(df_variants.dtypes)


#example row
#Uploaded_variation	Allele	Gene	Feature	Consequence	Existing_variation	SYMBOL	CANONICAL	SIFT	PolyPhen	HGVSc	HGVSp	AF	gnomADe_AF	CLIN_SIG	REVEL	SpliceAI_pred	DANN_score	MetaLR_score	CADD_raw_rankscore	ExAC_AF	ALFA_Total_AF	AlphaMissense_score	AlphaMissense_pred	turkishvariome_TV_AF
#2205837	G	ENSG00000186092	ENST00000641515	missense_variant	rs781394307	OR4F5	YES	tolerated(0.08)	benign(0.007)	ENST00000641515.2:c.107A>G	ENSP00000493376.2:p.Glu36Gly	-	0.02667	likely_benign	0.075	OR4F5|0.00|0.00|0.02|0.03|45|-1|-19|-1	0.95777141322514492	0.0013	0.24798	1.655e-03	0.0011802394199966278	0.0848,0.0854	B,B	-

#

# print 123..10





