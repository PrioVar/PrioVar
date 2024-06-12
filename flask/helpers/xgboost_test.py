import pickle
import random
import xgboost as xgb
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from helpers import hpo_sample
from xgboost_train import read_embedding, read_dicts, HPO_SAMPLE_STRATEGIES
from helpers.gene_mapping import get_gene_phenotype_relations_and_frequency
import matplotlib.pyplot as plt


# from hpo_sample create instance of Network
hpo_network = hpo_sample.Network()

# from train1.py
max_number_of_relevant_phenotypes = max([sum(strategy[:2]) for strategy in HPO_SAMPLE_STRATEGIES])

# read dictionaries by using the functions in train1.py
gene_dict, hpo_dict = read_dicts()

# read the embeddings
node_embeddings = read_embedding()


# df_variants: dataframe of variants
# numof_pathogenic_variants: number of pathogenic variants to add to the patient
# numof_nonpathogenic_variants: number of non-pathogenic variants to add to the patient
# phenotype_sample_strategy: strategy to sample the phenotypes of the patient: precise, imprecise, noisy
# return: a list of variants and the phenotypes of the patient
def simulate_a_patient(
    df_variants,
    numof_pathogenic_variants,
    numof_nonpathogenic_variants,
    phenotype_sample_strategy
):

    # first randomly  sample A pathogenic variant (according to CLIN_SIG  classification)
    pathogenic_variants = df_variants[df_variants['CLIN_SIG'] == 'pathogenic']

    # randomly sample numof_pathogenic_variants pathogenic variants
    pathogenic_variants = pathogenic_variants.sample(n=numof_pathogenic_variants)

    # choose one of them as the target variant
    target_variant = pathogenic_variants.sample(n=1)

    # get the gene of the target variant
    target_gene = target_variant['SYMBOL']

    # get the string value of the gene without the index
    target_gene = target_gene.values[0].split('.')[0]

    # get the hpo ids of the target gene with frequency
    gene_phenotype_relations = get_gene_phenotype_relations_and_frequency()

    hpo_terms_and_frequencies = [
        relation[1:] for relation in gene_phenotype_relations if relation[0] == target_gene
    ]
    hpo_terms_and_frequencies = sorted(hpo_terms_and_frequencies,
                                       key=lambda x: x[1] if x[1] is not None else 0,
                                       reverse=True)
    hpo_ids = [relation[0] for relation in hpo_terms_and_frequencies[:max_number_of_relevant_phenotypes+1]]

    while len(hpo_ids) == 0 or target_gene not in gene_dict:
        target_variant = pathogenic_variants.sample(n=1)
        target_gene = target_variant['SYMBOL']
        target_gene = target_gene.values[0].split('.')[0]
        gene_phenotype_relations = get_gene_phenotype_relations_and_frequency()
        hpo_terms_and_frequencies = [relation[1:]
                                     for relation in gene_phenotype_relations
                                     if relation[0] == target_gene]
        hpo_terms_and_frequencies = sorted(hpo_terms_and_frequencies,
                                           key=lambda x: x[1] if x[1] is not None else 0,
                                           reverse=True)
        hpo_ids = [relation[0] for relation in hpo_terms_and_frequencies[:max_number_of_relevant_phenotypes+1]]

    # sample the phenotypes of the patient
    hpo_sample = hpo_network.sample_patient_phenotype_v2(
        hpo_ids,
        phenotype_sample_strategy[0],
        phenotype_sample_strategy[1],
        phenotype_sample_strategy[2]
    )

    # sample numof_nonpathogenic_variants non-pathogenic variants (which is either benign or likely benign)
    nonpathogenic_variants = df_variants[df_variants['CLIN_SIG'].isin(['benign', 'likely_benign'])]
    nonpathogenic_variants = nonpathogenic_variants.sample(n=numof_nonpathogenic_variants)

    # combine the pathogenic and non-pathogenic variants
    variants = pd.concat([pathogenic_variants, nonpathogenic_variants])

    #variants = variants.drop(columns=['CLIN_SIG'])
    # get hpo_sample_info from add_embedding_info_to_patient
    hpo_sample_info = add_embedding_info_to_patient(
        target_gene,
        hpo_sample,
        add_scaled_average_dot_product=True,
        add_scaled_min_dot_product=True,
        add_scaled_max_dot_product=True,
        add_average_dot_product=True,
        add_min_dot_product=True,
        add_max_dot_product=True
    )

    # add hpo_sample_info to the all variants ???
    variants = pd.concat([variants, pd.DataFrame([hpo_sample_info])], axis=1)

    return variants, hpo_sample, target_variant


def simulate_patients(df_variants, numof_pathogenic_variants, numof_nonpathogenic_variants, phenotype_sample_strategies, numof_patients, output_file):
     # simulate them and store the values so that they can be fed to the model and save
    patients = []
    for i in range(numof_patients):
        phenotype_sample_strategy = random.choice(phenotype_sample_strategies)
        patient = simulate_a_patient(df_variants, numof_pathogenic_variants,
                                     numof_nonpathogenic_variants, phenotype_sample_strategy)
        patients.append(patient)

    # save the patients as a pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(patients, f)

    return patients


def process_hpo_info(patients_file, output_file, hpo_processor_func):
    with open(patients_file, 'rb') as f:
        patients = pickle.load(f)

    # process the patients
    processed_patients = []
    for patient in patients:
        # get the hpo_sample of the patient
        hpo_sample = patient[1]

        # process hpo_sample of the patient add it to its variants df
        processed_hpo = hpo_processor_func(hpo_sample)
        patient[0] = pd.concat([patient[0], processed_hpo], axis=1)
        processed_patients.append(patient)

    # save the processed patients
    with open(output_file, 'wb') as f:
        pickle.dump(processed_patients, f)

    return processed_patients


# to evaluate xgboost model saved in model_file using the patients saved in patients_file
def evaluate_model(model_file, patients_file):
    # load the xgboost model from model_file path
    model = xgb.Booster()
    model.load_model(model_file)

    # load the patients
    with open(patients_file, 'rb') as f:
        patients = pickle.load(f)

    # for every patient
    # get the scores of model for every variant
    # sort the variants according to the scores
    # get the target variant's rank among the variants
    score_distribution = []  # involves '#Uploaded_variation', symbol, and the score

    for patient in patients:
        variants = patient[0]
        target_variant = patient[2]

        variants['HGVSp'] = variants['HGVSp'].astype('category')
        #variants = variants.drop(columns=['Gene', 'Existing_variation', 'CANONICAL', 'AlphaMissense_score', 'AlphaMissense_pred'])
        #variants = variants.drop(columns=['#Uploaded_variation'])
        variants[['DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL']] = variants[
            ['DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL']].replace('None', np.nan).astype(
            'float')

        X = variants.drop(columns=['CLIN_SIG'])

        # imputer: replace missing values with the mean of the column (only for numerical columns)
        # don't use it for categorical columns

        imputer = SimpleImputer(strategy='mean')
        # get the numerical columns
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        # fit the imputer to the training data
        imputer.fit(X[numerical_columns])

        # get the target row from X which has the target variant
        target_row_index = X[X['#Uploaded_variation'] == target_variant['#Uploaded_variation'].values[0]].index[0]

        # reorder the columns according to ['Allele', 'Feature', 'Consequence', 'SYMBOL', 'SIFT', 'PolyPhen', 'HGVSp', 'AF', 'gnomADe_AF', 'REVEL', 'DANN_score', 'MetaLR_score', 'CADD_raw_rankscore', 'ExAC_AF', 'ALFA_Total_AF', 'turkishvariome_TV_AF', 'HGVSc_number', 'HGVSc_change', 'HGVSp_number', 'HGVSp_change', 'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL', 'AlphaMissense_score_mean', 'AlphaMissense_std_dev', 'AlphaMissense_pred_A', 'AlphaMissense_pred_B', 'AlphaMissense_pred_P', 'PolyPhen_number', 'SIFT_number', 'scaled_average_dot_product', 'scaled_min_dot_product', 'scaled_max_dot_product', 'average_dot_product', 'min_dot_product', 'max_dot_product']
        X = X[['Allele', 'Feature', 'Consequence', 'SYMBOL', 'SIFT', 'PolyPhen', 'HGVSp', 'AF', 'gnomADe_AF', 'REVEL', 'DANN_score', 'MetaLR_score', 'CADD_raw_rankscore', 'ExAC_AF', 'ALFA_Total_AF', 'turkishvariome_TV_AF', 'HGVSc_number', 'HGVSc_change', 'HGVSp_number', 'HGVSp_change', 'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG', 'DP_AL', 'DP_DG', 'DP_DL', 'AlphaMissense_score_mean', 'AlphaMissense_std_dev', 'AlphaMissense_pred_A', 'AlphaMissense_pred_B', 'AlphaMissense_pred_P', 'PolyPhen_number', 'SIFT_number', 'scaled_average_dot_product', 'scaled_min_dot_product', 'scaled_max_dot_product', 'average_dot_product', 'min_dot_product', 'max_dot_product']]

        dtest = xgb.DMatrix(X, enable_categorical=True)

        # get the scores of model for every variant
        scores = model.predict(dtest)

        # scores is probability of 4 classes
        # score : p3 + p2 x 0.4 - p0 - p1 x 0.4
        scores = scores[:, 3] + scores[:, 2] * 0.6 - scores[:, 0] - scores[:, 1] * 0.6

        # sort the variants according to the scores
        variants['scores'] = scores
        #target_score = variants.loc[target_row_index, 'scores']
        variants = variants.sort_values(by='scores', ascending=False)

        variants["sorted_index"] = [i for i in range(len(variants))]

        # its rank among the sorted scores
        target_variant_rank = variants.loc[target_row_index, 'sorted_index']

        score_distribution.append(( target_variant_rank ,variants[['#Uploaded_variation', 'SYMBOL', 'scores']]))
        print(target_variant_rank, "out of", len(variants))

    return score_distribution


# process to add add_scaled_average_dot_product=True, add_scaled_min_dot_product=True, add_scaled_max_dot_product=True, add_average_dot_product=True, add_min_dot_product=True, add_max_dot_product=True


# returns a dictionary of related info
def add_embedding_info_to_patient(
        gene,
        sampled_hpo,
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
    global node_embeddings, gene_dict, hpo_dict

    if add_fix_num_phen_embedding > 0:
        pass

    hpo_embedding_info = {}

    # get the gene embedding
    gene_embedding = node_embeddings[gene_dict[gene]]

    # get the embeddings of the HPO terms
    hpo_embeddings = [node_embeddings[hpo_dict[hpo]] for hpo in sampled_hpo]

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


"""
df_variants = add_embedding_info(df_variants, path_to_embedding='../data/node_embeddings.txt', path_to_gene_dict='../data/gene_dict.pkl', path_to_hpo_dict='../data/hpo_dict.pkl', path_to_sampled_hpo='../data/sampled_hpoIDs__with_freq_for_variants.pkl',
                                    add_scaled_average_dot_product=True, add_scaled_min_dot_product=True, add_scaled_max_dot_product=True, add_average_dot_product=True, add_min_dot_product=True, add_max_dot_product=True)
"""

# sample patients

# read the variants variants_cleaned.pkl
print('Doing the simulation')
#df_variants = pd.read_pickle('../data/variants_cleaned.pkl')
#simulate_patients(df_variants, 10, 10000, hpo_strategies, 100, '../data/patients2.pkl')

#'../data/model_with_first_embedding_freq_based.xgb'
scores = evaluate_model('../data/model_with_first_embedding_freq_based.xgb', '../data/patients2.pkl')

# visualize the scores
ranks = [score[0] for score in scores]

print('Mean rank:', np.mean(ranks))

plt.hist(ranks, bins=100)
plt.xlabel('Rank of the target variant')
plt.ylabel('Frequency')
plt.title('Distribution of the ranks of the target variant')
plt.show()

# sort and save the ranks under ../data
ranks = sorted(ranks)
with open('../data/ranks.txt', 'w') as f:
    for rank in ranks:
        f.write(str(rank) + '\n')  # write the rank to the file

a = 5
