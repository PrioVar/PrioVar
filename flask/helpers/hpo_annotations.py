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
        #Skip the header
        for i in range(4):
            next(file)
        
        reader = csv.DictReader(file, delimiter='\t')
        #Limit the number of rows for testing to 500
        reader = itertools.islice(reader, 500)
        for row in reader:
            # Create disease node if not exists
            create_disease_query = (
                "MERGE (d:Disease {name: $disease_name}) "
            )
            session.run(create_disease_query,  disease_name=row['disease_name'])
            # Create relationship with phenotype term
            create_relationship_query = (
                "MATCH (d:Disease {name: $disease_name}), (p:PhenotypeTerm {id: $hpo_id}) "
                "MERGE (d)-[r:ASSOCIATED_WITH_PHENOTYPE]->(p) "
                "SET r.frequency = $frequency, r.database_id = $database_id "
            )
            hpo_idd = int(row['hpo_id'][-7:])
            session.run(create_relationship_query, disease_name=row['disease_name'], hpo_id=hpo_idd, frequency=row['frequency'], database_id=row['database_id'])


# Function to connect to the database and create the nodes and relationships
def initiate_disease_database():
    driver = create_session(uri, username, password)
    with driver.session() as session:
        read_and_create_diseases_from_hpoa(session)
    driver.close()