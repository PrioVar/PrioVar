from bs4 import BeautifulSoup

print('start...')

with open('helpers/data/orphanet_disease_to_gene.xml', 'r', encoding='ISO-8859-1') as f:
    data = f.read()

disease_to_gene = BeautifulSoup(data, 'xml')

with open('helpers/data/orphanet_disease_to_phenotype.xml', 'r', encoding='ISO-8859-1') as f:
    data = f.read()

disease_to_phenotype = BeautifulSoup(data, 'xml')

print('end.')
i = 0
# Iterate through Disorder elements
for disorder in disease_to_gene.find_all('Disorder'):
    # Extract information from each Disorder element
    disease_id = disorder['id']
    orpha_code = disorder.find('OrphaCode').text
    disease_name = disorder.find('Name', {'lang': 'en'}).text

    # Extract information from DisorderGeneAssociation element
    gene_association = disorder.find('DisorderGeneAssociation')
    source_of_validation = gene_association.find('SourceOfValidation').text
    gene_name = gene_association.find('Gene').find('Name', {'lang': 'en'}).text
    gene_symbol = gene_association.find('Gene').find('Symbol').text

    # Print the information
    print(f"Disease ID: {disease_id}")
    print(f"Orpha Code: {orpha_code}")
    print(f"Disease Name: {disease_name}")
    print(f"Source of Validation: {source_of_validation}")
    print(f"Gene Name: {gene_name}")
    print(f"Gene Symbol: {gene_symbol}")
    print("--------------")
    i = i + 1

print("disease to gene", i)

#Iterate through disease_to_phenotype elements
i = 0
j = 0
for disorder in disease_to_phenotype.find_all('Disorder'):
    # Extract information from each Disorder element
    disease_id = disorder['id']
    orpha_code = disorder.find('OrphaCode').text
    disease_name = disorder.find('Name', {'lang': 'en'}).text

    # Extract information from DisorderPhenotypeAssociation element
    phenotype_association_list = disorder.find('HPODisorderAssociationList')

    i = i + 1

    if phenotype_association_list:
        for phenotype_association in phenotype_association_list.find_all('HPODisorderAssociation'):
            # Extract information from HPODisorderAssociation element
            phenotype_id = phenotype_association.find('HPO').find('HPOId').text
            phenotype_name = phenotype_association.find('HPO').find('HPOTerm').text
            frequency = phenotype_association.find('HPOFrequency').find('Name', {'lang': 'en'}).text

            # Print the information
            print(f"Disease ID: {disease_id}")
            print(f"Orpha Code: {orpha_code}")
            print(f"Disease Name: {disease_name}")
            print(f"Phenotype ID: {phenotype_id}")
            print(f"Phenotype Name: {phenotype_name}")
            print(f"Frequency: {frequency}")
            print("--------------")
            j = j + 1

print("In disease_to_phenotype:\n")
print("disease elements: ", i)
print("# of disorder_to_phenotype relations: ", j)