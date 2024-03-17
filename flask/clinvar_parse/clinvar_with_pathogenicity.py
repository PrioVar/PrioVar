from tqdm import tqdm

def create_dictionary_from_file(input_file_path):
    variant_dict = {}
    print("Creating hash table for variants...")
    with open(input_file_path, 'r') as file:
        for line in tqdm(file):
            if line.startswith('#'):
                continue  # Skip comment lines
            columns = line.strip().split('\t')
            info = columns[7].split(';')
            pathogenicity_info = [x for x in info if x.startswith('CLNSIG=')]
            pathogenicity_info = pathogenicity_info[0].split('=')[1]
            if pathogenicity_info == 'Pathogenic' or pathogenicity_info == 'Likely_pathogenic' or pathogenicity_info == 'Pathogenic/Likely_pathogenic' or pathogenicity_info == 'Pathogenic/Likely_pathogenic,Conflicting_interpretations_of_pathogenicity' or pathogenicity_info == 'Pathogenic,Conflicting_interpretations_of_pathogenicity' or pathogenicity_info == 'Pathogenic,association' or pathogenicity_info == 'Pathogenic,association,Conflicting_interpretations_of_pathogenicity' or pathogenicity_info == 'Pathogenic,Conflicting_interpretations_of_pathogenicity,association' or pathogenicity_info == 'Pathogenic,association,drug_response' or pathogenicity_info == 'Pathogenic,association,Conflicting_interpretations_of_pathogenicity,drug_response' or pathogenicity_info == 'Pathogenic,Conflicting_interpretations_of_pathogenicity,association,drug_response' or pathogenicity_info == 'Pathogenic,association,Conflicting_interpretations_of_pathogenicity,drug_response,other' or pathogenicity_info == 'Pathogenic,association,Conflicting_interpretations_of_pathogenicity,other' or pathogenicity_info == 'Pathogenic,association,drug_response,other' or pathogenicity_info == 'Pathogenic,Conflicting_interpretations_of_pathogenicity,association,drug_response,other' or pathogenicity_info == 'Pathogenic,Conflicting_interpretations_of_pathogenicity,association,other' or pathogenicity_info == 'Pathogenic,association,drug_response,other' or pathogenicity_info == 'Pathogenic,Conflicting_interpretations_of_pathogenicity,association,drug_response,other' or pathogenicity_info == 'Pathogenic,association,Conflicting_interpretations_of_pathogenicity,drug_response,other' or pathogenicity_info == 'Pathogenic,association,Conflicting_interpretations_of_pathogenicity,other' or pathogenicity_info == 'Pathogenic,association,drug_response,other' or pathogenicity_info == 'Pathogenic,Conflicting_interpretations_of_pathogenicity,association,drug_response,other' or pathogenicity_info == 'Pathogenic,Conflicting_interpretations_of_pathogenicity,association,other' or pathogenicity_info == 'Pathogenic,association,drug_response,other' or pathogenicity_info == 'Benign':
                key = (str(columns[0]), str(columns[1]), str(columns[3]), str(columns[4]))  # Use columns 0, 1, 3, and 4 as the key
                # Assuming the pathogenicity value is in the 10th column (index 9)
                variant_dict[key] = None
    print("Hash table created!")
    return variant_dict

def add_pathogenicities(pathogenicity_file_path, variant_dict):
    num_of_matched = 0
    print("Adding pathogenicity information...")
    with open(pathogenicity_file_path, 'r') as file:
        for line in tqdm(file):
            if line.startswith('#'):
                continue  # Skip comment lines
            columns = line.strip().split('\t')
            key = (columns[0][3:], columns[1], columns[2], columns[3])  # Use columns 0, 1, 2, and 3 as the key
            if key in variant_dict:
                variant_dict[key] = columns[4]  # Store the pathogenicity and pathogenicity class
                num_of_matched += 1

    print("Pathogenicity information added!")
    return num_of_matched

def write_hashed_values(output_file, variant_dict):
    print("Writing hashed values to file...")
    with open(output_file, 'w') as output_file:
        output_file.write("#CHROM\tPOS\tREF\tALT\tAM_PATHOGENICITY\n")
        for key in tqdm(variant_dict.keys()):
            # Compute the hash value for the pair of key and value
            value = variant_dict[key]
            # Write the key, value, and hash value to the output file
            output_file.write(f"{key[0]}\t{key[1]}\t{key[2]}\t{key[3]}\t{value}\n")
    print("Hashed values written to file!")

# Example usage:
input_file_path = 'clinvar_20240127.vcf'
pathogenicity_file_path = 'AlphaMissense_variants_pathogenicity.vcf'
output_file_path = 'clinvar_20240127_with_alphamissense_pathogenicity.vcf'

variant_dict = create_dictionary_from_file(input_file_path)
num_of_matched = add_pathogenicities(pathogenicity_file_path, variant_dict)
print("Number of matched variants: ", num_of_matched)
write_hashed_values(output_file_path, variant_dict)