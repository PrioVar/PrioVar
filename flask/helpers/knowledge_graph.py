import os
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from config import uri, username, password, API_KEY

os.environ["OPENAI_API_KEY"] = API_KEY

graph = Neo4jGraph(
    url=uri, username=username, password=password
)

chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125"),
    qa_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125"),
    verbose=True,
    validate_cypher=True,
)


def get_answer(question: str):
    return chain.invoke(" " + question + " ")

