import json
from neo4j import GraphDatabase
from os import path
from config import uri, username, password

def read_hpo_from_json():
    with open(path.join('data', 'hp.json'), 'r') as file:
        data = json.load(file)

    graphs = data["graphs"]
    graph = graphs[0]
    graph_id = graph["id"]
    meta = graph["meta"]
    nodes = graph["nodes"]
    edges = graph["edges"]
    property_chain_axioms = graph["propertyChainAxioms"]

    return graph_id, meta, nodes, edges, property_chain_axioms


def process_nodes(nodes):
    items = [] # to store the nodes as dictionaries with their properties: id, name, definition, comment, synonyms, xrefs,

    for node in nodes:
        #if its type is class
        if  "type"  in node.keys() and node["type"] == "CLASS":
            # create dictionary
            item = {}
            # add id by getting the last 7 chars of the id
            item["id"] = int( node["id"][-7:])
            # add name
            if "lbl" in node.keys():
                item["name"] = node["lbl"]
            # add definition
            if "meta" in node.keys():
                #check if deprecated
                if "deprecated" in node["meta"].keys() and node["meta"]["deprecated"]:
                    continue

                if "definition" in node["meta"].keys():
                    item["definition"] = node["meta"]["definition"]["val"]
                else:
                    item["definition"] = ""
                # add comment
                if "comments" in node["meta"].keys():
                    item["comment"] = node["meta"]["comments"][0]
                else:
                    item["comment"] = ""

                # add synonyms by getting "val"'s from the list of dictionaries
                if "synonyms" in node["meta"].keys():
                    item["synonyms"]= [synonym["val"] for synonym in node["meta"]["synonyms"]]
                else:
                    item["synonyms"] = []

                # add xrefs
                if "xrefs" in node["meta"].keys():
                    item["xrefs"] = [xref["val"] for xref in node["meta"]["xrefs"]]
                else:
                    item["xrefs"] = []
            else:
                item["definition"] = ""
                item["comment"] = ""
                item["synonyms"] = []
                item["xrefs"] = []

            # add item to items
            items.append(item)
    return items

def process_edges(edges):
    edge_items = []

    for edge in edges:
        edge = [int(edge["sub"][-7:]), int(edge["obj"][-7:])]
        edge_items.append(edge)
    return edge_items


def save_nodes(items):

    # Function to add an Phenotype term (node) to Neo4j and id is the primary key
    def add_item(tx, PhenotypeTerm):
        tx.run("MERGE (a:PhenotypeTerm {id: $id, name: $name, definition: $definition, comment: $comment, synonyms: $synonyms, xrefs: $xrefs})",
               id=PhenotypeTerm["id"], name=PhenotypeTerm["name"], definition=PhenotypeTerm["definition"], comment=PhenotypeTerm["comment"], synonyms=PhenotypeTerm["synonyms"], xrefs=PhenotypeTerm["xrefs"])

    # Function to create a constraint on the id property
    def create_constraint(tx):
        constraints = tx.run("SHOW CONSTRAINTS").data()
        if not any(constraint['name'] == 'constraint_phenotypeterm_id' for constraint in constraints):
            tx.run("CREATE CONSTRAINT constraint_phenotypeterm_id FOR (n:PhenotypeTerm) REQUIRE n.id IS UNIQUE")

    # Initialize the Neo4j driver
    driver = GraphDatabase.driver(uri, auth=(username, password))

    # Insert data into Neo4j
    with driver.session() as session:
        session.write_transaction(create_constraint)

        for item in items:
            session.write_transaction(add_item, item)

def save_edges(edge_items):

    # Function to connect 2 phenotypes
    def add_relationship(tx, item_id1, item_id2):
        tx.run("MATCH (a:PhenotypeTerm {id: $id1}), (b:PhenotypeTerm {id: $id2}) "
               "MERGE (a)-[:IS_A]->(b)", id1=item_id1, id2=item_id2)

    # Initialize the Neo4j driver
    driver = GraphDatabase.driver(uri, auth=(username, password))

    # Insert data into Neo4j
    with driver.session() as session:
        #session.write_transaction(create_constraint)

        for item in edge_items:
            session.write_transaction(add_relationship, item[0], item[1])

#a = read_hpo_from_json()
#b = 5