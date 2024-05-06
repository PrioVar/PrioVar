from flask import Flask, request
from flask_cors import CORS

from helpers.hpo import read_hpo_from_json, process_nodes, process_edges, save_nodes, save_edges
from helpers.clinvar import read_clinvar, save_clinvar
from helpers.hpo_annotations import initiate_disease_database, initiate_gene_database
from helpers.annotation import annotate_variants, get_all_annotated_variants
from helpers.knowledge_graph import get_answer
from helpers.ClinicalResearchAssistant import analyze
from helpers.file_decode import read_file_content_and_return_df
from config import api_username, api_password, api_auth_token
import requests
import json
from neo4j import GraphDatabase
from os import path
from config import uri, username, password


def api_save_vcf_file(file, filename = "tinyy.vcf"):
    '''
        # now, send a request to lidyagenomics.com/libra/api/v1/vcf/cs492upload
        # keep the same headers as above
        # also, request.data["file"] should be the file to be uploaded, which is data/tinyy.vcf
        # Request should include a Content - Disposition header with a filename parameter
        response2 = requests.post("http://lidyagenomics.com/libra/api/v1/vcf/cs492upload", headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0",
            "Accept": "application/json, text/plain, /",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Authorization": f"Token {api_auth_token}",
            "Connection": "keep-alive",
            "Referer": "http://lidyagenomics.com/libra/files",
            "Cookie": "csrftoken=ul82ceOrSl2g2fO1VcC8tcJWJ54TYI5j7qRf4tcKkhadhifSNN2WkOckyKQCD7B1",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Content-Disposition": "attachment; filename=tinyy.vcf"
        }, files=files)

    files['file'][1].close()
    print(response2.json())
    print(response2)'''

    files = {'file': (filename, file)}

    response1 = requests.post("http://lidyagenomics.com/libra/api/v1/vcf/cs492upload", headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0",
        "Accept": "application/json, text/plain, /",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Authorization": f"Token {api_auth_token}",
        "Connection": "keep-alive",
        "Referer": "http://lidyagenomics.com/libra/files",
        "Cookie": "csrftoken=ul82ceOrSl2g2fO1VcC8tcJWJ54TYI5j7qRf4tcKkhadhifSNN2WkOckyKQCD7B1",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Content-Disposition": f"attachment; filename={filename}"
    }, files=files)

    files['file'][1].close()
    print("api_save_vcf_file:")
    print(response1.json())
    print(response1)

    return response1.json()


#'vcf/cs492annotate/<str:vcf_id>'  bu istedigimiz islemi baslatmaya yariyor.
#endpointten de anlasilacagi uzere <str:vcf_id> yerine onceki stepten donulen id'yi koyacaksiniz.
# bu basladiktan sonra biraz surecek islem ama bu kismi ben handleladim ucuncu stepte.
# ucuncu endpointi okurken anlarsiniz
#
def api_start_analysis(vcf_id):

    response2 = requests.post(f"http://lidyagenomics.com/libra/api/v1/vcf/cs492annotate/{vcf_id}", headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0",
        "Accept": "application/json, text/plain, /",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Authorization": f"Token {api_auth_token}",
        "Connection": "keep-alive",
        "Referer": "http://lidyagenomics.com/libra/files",
        "Cookie": "csrftoken=ul82ceOrSl2g2fO1VcC8tcJWJ54TYI5j7qRf4tcKkhadhifSNN2WkOckyKQCD7B1",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin"
    })

    print("api_start_analysis:")
    #print(response2.json())
    print(response2)
    return response2

#vcf/cs492getoutput/<str:filename>' bu da size output donduruyor iste.
# Eger bu id'de bi vcf yoksa size bunu soyluyor.
# Eger varsa ama step 2 bitmemisse sunu donuyor "Response({"error": "There is such vcf but the job is not finished yet"}, status=201)".
# Eger bitmisse de dogrudan output donuyor. bitmemisse 201 olarak yapmisim simdi fark ettim xdd debuglamasi kolay olsun diye yaptiydim.
# Siz onu halledersiniz artik kendiniz.

def api_get_output(vcf_id):
    response3 = requests.get(f"http://lidyagenomics.com/libra/api/v1/vcf/cs492getoutput/{vcf_id}", headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:124.0) Gecko/20100101 Firefox/124.0",
        "Accept": "application/json, text/plain, /",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Authorization": f"Token {api_auth_token}",
        "Connection": "keep-alive",
        "Referer": "http://lidyagenomics.com/libra/files",
        "Cookie": "csrftoken=ul82ceOrSl2g2fO1VcC8tcJWJ54TYI5j7qRf4tcKkhadhifSNN2WkOckyKQCD7B1",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin"
    })
    print("api_get_output:")
    #print(response3.json())
    print(response3)
    return response3


def update_vcf_file(tx, file_id, new_api_file_id, new_file_status):
    query = """
    MATCH (f:VCFFile {id: $id})
    SET f.api_file_id = $api_file_id, f.fileStatus = $file_status
    """
    tx.run(query, id=file_id, api_file_id=new_api_file_id, file_status=new_file_status)

def set_vcf_file_details(file_id, new_api_file_id, new_file_status):
    # Initialize the Neo4j driver
    driver = GraphDatabase.driver(uri, auth=(username, password))

    # Update the specific VCFFile node in Neo4j
    with driver.session() as session:
        session.write_transaction(update_vcf_file, file_id, new_api_file_id, new_file_status)

