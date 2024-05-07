from neo4j import GraphDatabase
from os import path
from config import uri, username, password
import csv 
import itertools


def create_session(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver


# Function to read the HPOA file and create the disease nodes and relationships with phenotype terms
def read_and_create_diseases_from_hpoa(session):
    with open(path.join('data', 'phenotype.hpoa'), 'r') as file:
        # Skip the header
        for i in range(4):
            next(file)
        
        reader = csv.DictReader(file, delimiter='\t')
        # Limit the number of rows for testing to 500
        reader = itertools.islice(reader, 2500)
        for row in reader:
            # Create disease node if not exists
            create_disease_query = (
                "MERGE (d:Disease {diseaseName: $diseaseName}) "
            )
            session.run(create_disease_query,  diseaseName=row['disease_name'])
            # Create relationship with phenotype term
            create_relationship_query = (
                "MATCH (d:Disease {diseaseName: $diseaseName}), (p:PhenotypeTerm {id: $hpo_id}) "
                "MERGE (d)-[r:DISEASE_ASSOCIATED_WITH_PHENOTYPE]->(p) "
                "SET r.frequency = $frequency, r.databaseId = $databaseId "
            )
            hpo_idd = int(row['hpo_id'][-7:])
            session.run(create_relationship_query, diseaseName=row['disease_name'], hpo_id=hpo_idd, frequency=row['frequency'], databaseId=row['database_id'])


# Function to read genes_to_phenotype.txt and create the gene nodes and relationships with phenotype terms
def read_and_create_genes_from_genes_to_phenotype(session):
    with open(path.join('data', 'genes_to_phenotype.txt'), 'r') as file:
        #Get the first line as column names
        reader = csv.DictReader(file, delimiter='\t')
        #Limit the number of rows for testing to 500
        reader = itertools.islice(reader, 2500)
        for row in reader:
            # Create gene node if not exists
            create_gene_query = (
                "MERGE (g:Gene {geneSymbol: $gene_symbol}) "
            )
            session.run(create_gene_query,  gene_symbol=row['gene_symbol'])
            # Create relationship with phenotype term
            create_relationship_query = (
                "MATCH (g:Gene {geneSymbol: $gene_symbol}), (p:PhenotypeTerm {id: $hpo_id}) "
                "MERGE (g)-[:GENE_ASSOCIATED_WITH_PHENOTYPE]->(p) "
            )
            hpo_idd = int(row['hpo_id'][-7:])
            session.run(create_relationship_query, gene_symbol=row['gene_symbol'], hpo_id=hpo_idd)


# Function to connect to the database and create the disease nodes and relationships
def initiate_disease_database():
    driver = create_session(uri, username, password)
    with driver.session() as session:
        read_and_create_diseases_from_hpoa(session)
    driver.close()


# Function to connect to the database and create the gene nodes and relationships
def initiate_gene_database():
    driver = create_session(uri, username, password)
    with driver.session() as session:
        read_and_create_genes_from_genes_to_phenotype(session)
    driver.close()