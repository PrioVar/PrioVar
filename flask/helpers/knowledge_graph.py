import os
from neo4j import GraphDatabase
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from config import uri, username, password, API_KEY


class KnowledgeGraphQA:
    def __init__(self):
        """
        Initializes the Neo4j graph connection, the QA chain, and the Neo4j driver.
        """
        # Set the OpenAI API key
        os.environ["OPENAI_API_KEY"] = API_KEY

        # Initialize the graph
        self.graph = Neo4jGraph(
            url=uri, username=username, password=password
        )

        # Set up the QA chain with specific configurations
        self.chain = GraphCypherQAChain.from_llm(
            graph=self.graph,
            cypher_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125"),
            qa_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125"),
            verbose=True,
            validate_cypher=True,
        )

        # Initialize the Neo4j driver
        self.driver = GraphDatabase.driver(uri, auth=(username, password))


def get_answer(question: str) -> str:
    """
    Retrieves the answer by querying the knowledge graph using the provided question.

    :param kg_qa: An instance of KnowledgeGraphQA for accessing the graph and chain
    :param question: Question from the user
    :return: Answer as a string
    """
    kg_qa = KnowledgeGraphQA()
    answer = kg_qa.chain.invoke(" " + question + " ")
    return answer
