import os
from flask import jsonify
from Bio import Entrez
import re
from Bio import Medline
from openai import OpenAI
from transformers import BertTokenizer, BertModel
from config import API_KEY, uri, username, password
from neo4j import GraphDatabase
from datetime import datetime, timezone, timedelta
import torch
import faiss
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class ClinicalResearchAssistant:
    def __init__(self):
        # Open AI API Key
        os.environ["OPENAI_API_KEY"] = API_KEY

        # Configure your email so that the NCBI service knows who you are
        Entrez.email = "dummy@dummy.com"

        # Define the ChatGPT model
        self.client = OpenAI()
        self.model_name = "gpt-3.5-turbo-0125"
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        # Close the Neo4j connection
        self.driver.close()

    def chat(self, message):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": f"{message}"},
            ],
            temperature=0.1
        )
        return response.choices[0].message.content

    def get_mesh_terms(self, term, variable):
        handle = Entrez.esearch(db="mesh", term=term)
        record = Entrez.read(handle)
        handle.close()
        mesh_terms = []
        for translation in record['TranslationSet']:
            terms = translation['To'].split(' OR ')
            for term in terms:
                if '[MeSH Terms]' in term:
                    mesh_terms.append(term.replace('[MeSH Terms]', '').replace('"', '').strip())
        if variable == 'Patient':
            query = " AND ".join([f"{term}" for term in mesh_terms])
        else:
            query = " OR ".join([f"{term}" for term in mesh_terms])
        print(variable, " Query: ")
        print(query)
        print("\n")
        return query

    def construct_query(self, pico_variables):
        p_query = self.get_mesh_terms(pico_variables['Patient'], 'Patient')
        i_query = self.get_mesh_terms(pico_variables['Intervention'], 'Intervention')
        c_query = self.get_mesh_terms(pico_variables['Comparison'], 'Comparison')
        o_query = self.get_mesh_terms(pico_variables['Outcome'], 'Outcome')
        return f"({p_query}) AND ({i_query}) AND ({c_query}) AND ({o_query})"

    def perform_article_search(self, query):
        handle = Entrez.esearch(db="pubmed", term=query)
        record = Entrez.read(handle)
        handle.close()
        return record['IdList']

    def fetch_article_details(self, idlist):
        handle = Entrez.efetch(db="pubmed", id=idlist, rettype="medline", retmode="text")
        records = Medline.parse(handle)
        articles = list(records)
        handle.close()
        return articles

    def embed_text(self, text):
        if not text:
            return None
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs['pooler_output'].numpy()

    def search_vector_database(self, pico_clinical_question, vectors, articles):
        query_vector = self.embed_text(pico_clinical_question)
        if query_vector is None:
            return []
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        k = 5 if len(vectors) >= 5 else len(vectors)
        D, I = index.search(query_vector, k)
        return [articles[i] for i in I[0]]

    def create_chat_in_db(self, medical_center_id, message, response_data):
        with self.driver.session() as session:
            result, timestamp = session.write_transaction(
                self._create_and_link_chat, medical_center_id, message, response_data
            )
            return result, timestamp

    @staticmethod
    def _create_and_link_chat(tx, medical_center_id, question, response_data):
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        utc_plus_3 = timezone(timedelta(hours=3))
        timestamp = utc_now.astimezone(utc_plus_3).isoformat()
        query = (
            "MATCH (mc:MedicalCenter) WHERE ID(mc) = $medical_center_id "
            "CREATE (c:Chat {question: $question, timestamp: $timestamp, "
            "pico_clinical_question: $pico_clinical_question, "
            "article_count: $article_count, "
            "article_titles: $article_titles, "
            "RAG_GPT_output: $RAG_GPT_output}) "
            "MERGE (c)-[:HAS_CHAT]->(mc) "
            "RETURN c"
        )
        parameters = {
            "medical_center_id": int(medical_center_id),
            "question": question,
            "timestamp": timestamp,
            "pico_clinical_question": response_data.get('pico_clinical_question', ''),
            "article_count": response_data.get('article_count', 0),
            "article_titles": response_data.get('article_titles', []),
            "RAG_GPT_output": response_data.get('RAG_GPT_output', '')
        }
        result = tx.run(query, parameters)
        return result.single()[0], timestamp


def analyze(data):
    # Question that is to be asked by the clinician. Will be converted to PICO format by the ChatGPT
    clinical_question = data.get('clinical_question')
    medical_center_id = data.get('healthCenterId')
    if not clinical_question:
        return jsonify({'error': 'No clinical question provided'}), 400

    assistant = ClinicalResearchAssistant()
    print("Original Clinical Question: \n" + clinical_question)
    print("\n")

    # Rewrite the Clinical Question in PICO format
    pico_clinical_question = assistant.chat(
        'Rewrite the following clinical question below according to the PICO model using (P), (I) , (C), (O) notation to the right of the clause. For example: '
        + 'In patients with a fractured tibia (P) does surgical intervention (I) result in better outcomes compared to non-surgical management (C) in terms of healing time and functional recovery (O)?\n'
        + 'Above example is only for demonstrating format. Please rewrite the following clinical question in PICO format:'
        + clinical_question
    )
    print("PICO Clinical Question: \n" + pico_clinical_question)
    print("\n")

    # Parse the PICO question into its components
    # Regular expression to capture PICO components
    pico_string = pico_clinical_question
    # Refined regular expression to capture PICO components

    pattern = r"In (?P<Patient>.*?) \(P\), (?P<Intervention>.*?) \(I\) (?P<Comparison>.*?) \(C\) (?P<Outcome>.*?) \(O\)\?"

    match = re.match(pattern, pico_string)
    if match:
        pico_variables = match.groupdict()
    else:
        pico_variables = "No match found!"
    print("PICO Variables: ")
    print(pico_variables)
    print("\n")

    final_query = assistant.construct_query(pico_variables)
    print("Final Query: \n" + final_query + "\n")

    idlist = assistant.perform_article_search(final_query)
    print("length of idlist: " + str(len(idlist)) + "\n")

    articles = assistant.fetch_article_details(idlist)
    vectors = np.vstack([assistant.embed_text(article.get('AB', '')) for article in articles if article.get('AB')])
    nearest_articles = assistant.search_vector_database(pico_clinical_question, vectors, articles)

    # Print the nearest articles
    result = "\n"
    for article in nearest_articles:
        result += "Title: {}\nAbstract: {}\nJournal: {}\nAuthor: {}\nDate of publication: {}\nKeywords: {}\nMesh terms: {}\n\n".format(
            article.get("TI", "?"), article.get("AB", "?"), article.get("TA", "?"),
            ", ".join(article.get("AU", ["?"])), article.get("DP", "?"),
            ", ".join(article.get("OT", ["?"])), ", ".join(article.get("MH", ["?"]))
        )
    print(result)

    # Generate reports
    print("RAG GPT output:\n")
    research_res = assistant.chat(
        "Act as an evidence-based clinical researcher. Using only the following PubMed Abstracts to guide your content ("
        + result + "), create an evidence based medicine report that answers the following question: "
        + pico_clinical_question)
    print("Output:\n" + research_res + "\n")

    ''' print("Pure GPT output:\n")
    research_res_pure = assistant.chat(
        "Act as an evidence-based clinical researcher. Create an evidence based medicine report that answers the following question: "
        + pico_clinical_question + " Provide references to support your content.")
    print("Output:\n" + research_res_pure + "\n") '''
    print("Done")

    response_data = {
        'pico_clinical_question': pico_clinical_question,
        'article_count': len(nearest_articles),
        'article_titles': [article.get('TI', '?') for article in nearest_articles],
        'RAG_GPT_output': research_res,
    }

    result, timestamp = assistant.create_chat_in_db(medical_center_id, clinical_question, response_data)
    assistant.close()
    response_data['timestamp'] = timestamp

    return jsonify(response_data), 200


def get_chat_history(medical_center_id):
    assistant = ClinicalResearchAssistant()
    chats = assistant.get_chats_by_medical_center(medical_center_id)
    assistant.close()
    return jsonify(chats), 200
