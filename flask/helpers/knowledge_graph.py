import os
from neo4j import GraphDatabase
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from config import uri, username, password, API_KEY
from datetime import datetime, timedelta, timezone


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
            cypher_llm=ChatOpenAI(temperature=0, model="gpt-4-turbo-2024-04-09"),
            qa_llm=ChatOpenAI(temperature=0, model="gpt-4-turbo-2024-04-09"),
            verbose=True,
            validate_cypher=True,
        )

        # Initialize the Neo4j driver
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        # Close the Neo4j connection
        self.driver.close()

    def create_chat_in_db(self, medical_center_id, question, response):
        with self.driver.session() as session:
            result, timestamp = session.write_transaction(
                self._create_and_link_chat, medical_center_id, question, response
            )
            return result, timestamp

    @staticmethod
    def _create_and_link_chat(tx, medical_center_id, question, response):
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        utc_plus_3 = timezone(timedelta(hours=3))
        timestamp = utc_now.astimezone(utc_plus_3).isoformat()
        query = (
            "MATCH (mc:MedicalCenter) WHERE ID(mc) = $medical_center_id "
            "CREATE (c:GraphChat {question: $question, timestamp: $timestamp, "
            "response: $response}) "
            "MERGE (c)-[:HAS_GRAPH_CHAT]->(mc) "
            "RETURN c"
        )
        parameters = {
            "medical_center_id": int(medical_center_id),
            "question": question,
            "timestamp": timestamp,
            "response": response
        }
        result = tx.run(query, parameters)
        return result.single()[0], timestamp


def get_answer(question: str, medical_center_id: int) -> dict:
    """
    Retrieves the answer by querying the knowledge graph using the provided question.

    :param kg_qa: An instance of KnowledgeGraphQA for accessing the graph and chain
    :param question: Question from the user
    :return: Answer as a string
    """
    kg_qa = KnowledgeGraphQA()

    answer = kg_qa.chain.invoke(" " + question + " ")
    result, timestamp = kg_qa.create_chat_in_db(medical_center_id, question, answer['result'])
    kg_qa.close()

    return {"result": answer['result'], "timestamp": timestamp}
