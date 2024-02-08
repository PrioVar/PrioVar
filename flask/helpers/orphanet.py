from bs4 import BeautifulSoup

# Iterate through disease_to_phenotype elements
def get_disease_phenotype_relations():

    print('start...')
    with open('../data/orphanet_disease_to_phenotype.xml', 'r', encoding='ISO-8859-1') as f:
        data = f.read()
    disease_to_phenotype = BeautifulSoup(data, 'xml')
    print('end...')

    # store disease to phenotype relations
    disease_to_pheno = []

    for disorder in disease_to_phenotype.find_all('Disorder'):
        # Extract information from each Disorder element
        disease_id = disorder['id']
        orpha_code = disorder.find('OrphaCode').text
        disease_name = disorder.find('Name', {'lang': 'en'}).text

        # Extract information from DisorderPhenotypeAssociation element
        phenotype_association_list = disorder.find('HPODisorderAssociationList')

        if phenotype_association_list:
            for phenotype_association in phenotype_association_list.find_all('HPODisorderAssociation'):
                # Extract information from HPODisorderAssociation element
                phenotype_id = phenotype_association.find('HPO').find('HPOId').text
                phenotype_name = phenotype_association.find('HPO').find('HPOTerm').text
                frequency = phenotype_association.find('HPOFrequency').find('Name', {'lang': 'en'}).text
                #add disease to phenotype relation to dictionary
                disease_to_pheno.append([disease_id, phenotype_id, frequency])
    
    return disease_to_pheno

a = get_disease_phenotype_relations()