from os import path
import pandas as pd

NODE_DICTIONARY = {
    "HP:0040280": 1,
    "HP:0040281": 0.9,
    "HP:0040282": 0.55,
    "HP:0040283": 0.175,
    "HP:0040284": 0.03,
    "HP:0040285": 0,
}


def process_hpoa():
    """
    NOTE: AVERAGE OF ALL FREQUENCIES IS EQUAL TO 0.52,
    THEREFORE ALL THE VALUES ARE SCALED BY 10^-5 TO MAKE IT SUITABLE WITH THE
    REST OF THE VALUES IN THE GRAPH. (THE AVERAGE OF THE GENE-GENE INTERACTIONS
    IS ~5x10^-5)
    """
    # read phenotype.hpoa, skip the first 4 lines
    df = pd.read_csv(path.join('../data', 'phenotype.hpoa'), sep='\t', comment='#', skiprows=4)

    # get the last 7 characters of hpo_id and convert to int
    df["hpo_id"] = df["hpo_id"].str[-7:].astype(int)

    # get database_id, hpo_id, frequency
    df = df[["database_id", "hpo_id", "frequency", "disease_name"]]

    # for all rows where frequency is not null and contains 'HP:', map frequency using NODE_DICTIONARY
    df["frequency"] = df["frequency"].apply(
        lambda x: NODE_DICTIONARY[x] if x is not None and x in NODE_DICTIONARY else x)
    
    # Preprocess frequencies to handle division form and percentage form
    df["frequency"] = df["frequency"].apply(lambda x: eval(x.replace('%', '/100')) if isinstance(x, str) else x)

    # handle the nan values in frequency column by
    # replacing them with the mean of all non-null values
    mean = df[df["frequency"].notnull()]["frequency"].mean()
    df["frequency"] = df["frequency"].fillna(mean)

    # Convert frequencies to floats
    df["frequency"] = df["frequency"].astype(float)

    # scale the frequencies by 10^-5
    df["frequency"] = df["frequency"] * 10 ** -5

    # Group by disease_name, hpo_id, and database_id, and aggregate frequencies into a list
    disease_to_phenotype_all = (df.groupby('disease_name')
                                .agg({'hpo_id': list, 'database_id': list, 'frequency': list})
                                .reset_index())
    disease_database_ids = []

    # Iterate over the rows in the grouped DataFrame
    for index, row in disease_to_phenotype_all.iterrows():
        disease_name = row['disease_name']
        database_ids = list(set(row['database_id']))  # Extract unique database IDs
        database_ids_sorted = custom_sort(database_ids)  # Sort the database IDs
        disease_database_ids.append((disease_name, database_ids_sorted))

    return disease_to_phenotype_all, disease_database_ids


# Custom sorting function
def custom_sort(database_ids):
    order = {'OMIM': 0, 'ORPHA': 1, 'DECIPHER': 2}
    return sorted(database_ids, key=lambda x: order.get(x.split(':')[0], float('inf')))

a, b = process_hpoa()
#b = "1/5"
#b = '52\%'
#c = eval(b)
#a = process_hpoa()

# TODO: normally distribute these values 
# TODO: check if frequency is overwritten while iterating through the rows

#!!! A mixture of old code may be found below. The old code is not used in the current implementation. !!!

"""
# start reading phenotype.hpoa
 #eskikod =  read phenotype.hpoa line by line and find and find "Omim: " and "HP: " and store in list of tuples
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

    return list_of_tuples


def combine_omim_orpha_diseases(df) -> Dict:
    
    Map OMIM and ORPHA disease ids to disease names
    :param df:
    :return:
    
    # create a dictionary to map disease_id to disease_name
    disease_dict = {}
    for index, row in df.iterrows():
        disease_dict[row[0]] = row[1]

    return disease_dict


def proccess_hpoa():
    # read phenotype.hpoa skip the first 4 lines and store in dataframe

    df = pd.read_csv(path.join('../data', 'phenotype.hpoa'), sep='\t', comment='#', skiprows=4)

    # get the last 7 characters of hpo_id and convert to int
    df["hpo_id"] = df["hpo_id"].str[-7:].astype(int)

    # get database_id, hpo_id, frequency
    df = df[["database_id", "hpo_id", "frequency", "disease_name"]]

    # print all disease_names that are duplicated in datasets
    #print(df[df["disease_name"].duplicated()]["disease_name"].unique())

    # print all disease-to-phenotype relations that do not have frequency value
    #print(df[df["frequency"].isnull()])
    
    # Group by 'X' and filter out those groups with more than one unique entry in 'Y'
    grouped = df.groupby(['disease_name', 'hpo_id'])

    # Group by disease_name, hpo_id, and database_id, and aggregate frequencies into a list
    grouped = df.groupby(['disease_name', 'hpo_id', 'database_id'])['frequency'].agg(list).reset_index()

    # Further group by disease_name and merge rows
    merged = grouped.groupby('disease_name').agg({'hpo_id': 'unique', 'database_id': 'unique', 'frequency': 'sum'}).reset_index()

    print(merged)

    # Initialize an empty list to store the results
    result = []

    # Iterate over each group
    for name, group in grouped:
        # Extract disease_ids and frequencies for the current group
        disease_ids = group['database_id'].tolist()
        frequencies = group['frequency'].tolist()
        
        # Append the result as a tuple (disease_name, hpo_id, list of disease_ids, list of frequencies)
        result.append((name[0], name[1], disease_ids, frequencies))

    # Convert the result list to a DataFrame
    filtered_df = pd.DataFrame(result, columns=['disease_name', 'hpo_id', 'disease_ids', 'frequencies'])
    print(filtered_df)


    #old code starts here
    
    for group in grouped:
        print(group)
        break
    #filtered_groups = [group for name, group in grouped if group['database_id'].nunique() > 1 and group['frequency'].nunique() > 1]
    filtered_groups = [group for name, group in grouped if group['database_id'].nunique() >= 1 and group['frequency'].nunique() >= 1]
    # Concatenate the filtered groups back into a DataFrame
    filtered_df = pd.concat(filtered_groups)

    print(filtered_df)
    print("unique groups with different freq: ", filtered_df[['disease_name','hpo_id']].drop_duplicates().shape[0])
    
    # old code ends here
    # for all rows where frequency is not null and contains 'HP:', map frequency using NODE_DICTIONARY
    df["frequency"] = df["frequency"].apply(
        lambda x: NODE_DICTIONARY[x] if x is not None and x in NODE_DICTIONARY else x)

    # convert all rows with frequencies in the form of 'a/b' and 'a%' to float
    # df["frequency"] = df["frequency"].apply(lambda x: eval(x) if type(x) == str and  else x)

    # find the mean of all non-null values in frequency
    # frequency_list = df["frequency"].dropna().tolist()
    # frequency_list = list(map(float, frequency_list))
    # mean = sum(frequency_list) / len(frequency_list)

    disease_dict = combine_omim_orpha_diseases(df[["database_id", "disease_name"]])
    disease_name_frequency_list = []

    for index, row in df.iterrows():
        disease_name = disease_dict[row[0]]

        # Check if the disease name already exists in disease_name_freq_list
        found = False
        for item in disease_name_frequency_list:
            if item[0] == disease_name:
                item[1].append(row[2])  # Add the frequency to the existing frequency list
                found = True
                break
        if not found:
            # If the disease name is not found, add a new entry to disease_name_freq_list
            disease_name_frequency_list.append([disease_name, [row[2]]])



    # return list of tuples
    return df.values.tolist()
"""