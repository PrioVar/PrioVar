import math

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

driver = GraphDatabase.driver(uri, auth=(username, password))
def update_vcf_file(tx, file_id, new_api_file_id, new_file_status):
    query = """
    MATCH (f:VCFFile)
    WHERE ID(f) = $id
    SET f.api_file_id = $api_file_id, f.fileStatus = $file_status
    """
    tx.run(query, id=file_id, api_file_id=new_api_file_id, file_status=new_file_status)

def update_vcf_file_for_patient(tx, patient_id, new_api_file_id, new_file_status):

    # sets VCF for patient with the given patient_id, they are connected via "HAS_VCF_FILE" relationship
    query = """
    MATCH (p:Patient)-[:HAS_VCF_FILE]->(f:VCFFile)
    WHERE ID(p) = $patient_id
    SET f.api_file_id = $api_file_id, f.fileStatus = $file_status
    """
    tx.run(query, patient_id=patient_id, api_file_id=new_api_file_id, file_status=new_file_status)



def set_vcf_file_details(file_id, new_api_file_id, new_file_status):
    # Initialize the Neo4j driver
    #driver = GraphDatabase.driver(uri, auth=(username, password))

    # Update the specific VCFFile node in Neo4j
    with driver.session() as session:
        session.write_transaction(update_vcf_file, file_id, new_api_file_id, new_file_status)



def set_vcf_file_details_for_patient(patient_id, new_file_status, new_api_file_id = "none, yet"):


    with driver.session() as session:
        session.write_transaction(update_vcf_file_for_patient, patient_id, new_api_file_id, new_file_status)

def insert_variant(tx, variant_data, patient_id):
    # Create a new variant node or update it if it already exists

    # columns to add to the variant node:    private String allele;
    #     private String consequence;
    #     private String symbol;
    #     private String hgsvc_original;
    #     private String hgsvp_original;
    #     private String clin_sig;
    #     private String turkishvariome_tv_af_original;
    #     private Double priovar_score;
    #     private Double alphamissense_score_mean;
    # private String gene_original;

    """
     Index(['#Uploaded_variation', 'Allele', 'Gene', 'Feature', 'Consequence',
    //       'Existing_variation', 'SYMBOL', 'CANONICAL', 'SIFT', 'PolyPhen',
    //       'HGVSp', 'AF', 'gnomADe_AF', 'CLIN_SIG', 'REVEL', 'SpliceAI_pred',
    //       'DANN_score', 'MetaLR_score', 'CADD_raw_rankscore', 'ExAC_AF',
    //       'ALFA_Total_AF', 'AlphaMissense_score', 'AlphaMissense_pred',
    //       'turkishvariome_TV_AF', 'turkishvariome_TV_AF_original', 'HGSVc_original', 'HGSVp_original',
    //       'HGVSc_number', 'HGVSc_change', 'HGVSp_number', 'HGVSp_change',
    //       'SpliceAI_pred_symbol', 'DS_AG', 'DS_AL', 'DS_DG', 'DS_DL', 'DP_AG',
    //       'DP_AL', 'DP_DG', 'DP_DL', 'AlphaMissense_score_mean',
    //       'AlphaMissense_std_dev', 'AlphaMissense_pred_A', 'AlphaMissense_pred_B',
    //       'AlphaMissense_pred_P', 'PolyPhen_number', 'SIFT_number',
    //       'scaled_average_dot_product', 'scaled_min_dot_product',
    //       'scaled_max_dot_product', 'average_dot_product', 'min_dot_product',
    //       'max_dot_product', 'Priovar_score'],
    :param tx:
    :param variant_data:
    :param patient_id:
    :return:
    """

    columns_to_add = ['Uploaded_variation' ,'Allele', 'Consequence', 'SYMBOL', 'Priovar_score', 'Gene', 'turkishvariome_TV_AF_original', 'AlphaMissense_score_mean', 'HGSVc_original', 'HGSVp_original', 'CLIN_SIG']

    # delete the '#' character from the #uploaded_variation column name
    variant_data['Uploaded_variation'] = variant_data['#Uploaded_variation']
    # there are so much columns in the variant_data, so we need to filter them
    variant_data = {k: v for k, v in variant_data.items() if k in columns_to_add}

    # convert everything to string except Priovar_score, Uploaded_variation, and AlphaMissense_score_mean
    # if type is not string make it "Not Available"
    for key in variant_data:
        if key not in ['Priovar_score', 'Uploaded_variation', 'AlphaMissense_score_mean']:
            if not isinstance(variant_data[key], str):
                variant_data[key] = "-"

    # if alpha missense score mean is not float, make it "Not Available", if available, round it to 2 decimal points as string
    if not isinstance(variant_data['AlphaMissense_score_mean'], float):
        variant_data['AlphaMissense_score_mean'] = "-"
    # check if it is nan
    elif math.isnan(variant_data['AlphaMissense_score_mean']):
        variant_data['AlphaMissense_score_mean'] = "-"
    else:
        variant_data['AlphaMissense_score_mean'] = str(round(variant_data['AlphaMissense_score_mean'], 2))


    variant = tx.run(
        """
        MERGE (v:Variant {
            uploaded_variation: $Uploaded_variation
        })
        ON CREATE SET
            v.allele = $Allele,
            v.consequence = $Consequence,
            v.symbol = $SYMBOL,
            v.priovar_score = $Priovar_score,
            v.gene = $Gene,
            v.turkishvariome_tv_af_original = $turkishvariome_TV_AF_original,
            v.alpha_missense_score_mean = $AlphaMissense_score_mean,
            v.hgsvc_original = $HGSVc_original,
            v.hgsvp_original = $HGSVp_original,
            v.clin_sig = $CLIN_SIG
        RETURN v
        """, **variant_data).single().value()

    '''p_v = tx.run(
        """
        MATCH (p:Patient), (v:Variant {uploaded_variation: $uploaded_variation})
        WHERE ID(p) = $patient_id
        RETURN p, v
        """,
        uploaded_variation=variant_data['Uploaded_variation'],
        patient_id=patient_id
    )

    # print them to debug
    print(patient_id)
    print(variant_data['Uploaded_variation'])
    print(p_v)
    print("details of pv which is not subscriptable: ", p_v.single())


    v = tx.run("""
    MATCH(v: Variant {uploaded_variation: $uploaded_variation})
    RETURN v
    """, uploaded_variation=variant_data['Uploaded_variation']).single().value()

    print("v: ", v)

    # Check if the Patient exists
    p = tx.run("""
    MATCH(p: Patient)
    WHERE ID(p) = $patient_id
    RETURN p
    """, patient_id=patient_id).single()

    print("p: ", p)'''

    # Connect this variant to the patient
    tx.run(
        """
        MATCH (p:Patient), (v:Variant {uploaded_variation: $uploaded_variation})
        WHERE ID(p) = $patient_id
        MERGE (p)-[:HAS_VARIANT]->(v)
        """,
        uploaded_variation=variant_data['Uploaded_variation'],
        patient_id=patient_id
    )


    # add the gene_symbol if it does not exist
    geneSymbol = variant_data['SYMBOL']
    # check via math.isnan function

    # if gene symbol is not string, return
    if not isinstance(geneSymbol, str):
        print("Empty gene symbol")
        return




    #geneSymbol = "example_gene_symbol"

    tx.run(
        """
        MERGE (g:Gene {geneSymbol: $geneSymbol})
        """,
        geneSymbol=geneSymbol)


    # Connect this variant to the gene
    # "HAS_VARIANT_ON" relationship
    tx.run(
        """
        MATCH (v:Variant {uploaded_variation: $uploaded_variation}), (g:Gene {geneSymbol: $geneSymbol})
        MERGE (g)-[:HAS_VARIANT_ON]->(v)
        """,
        uploaded_variation=variant_data['Uploaded_variation'],
        geneSymbol=geneSymbol
    )
        

def get_patient_phenotypes(patient_id):

    # via "HAS_PHENOTYPE_TERM" relationship
    query = """ 
    MATCH (p:Patient)-[:HAS_PHENOTYPE_TERM]->(h:PhenotypeTerm)
    WHERE ID(p) = $patient_id
    RETURN h
    """

    with driver.session() as session:
        result = session.run(query, patient_id=patient_id)
        # if empty return [1], return as a list of integers
        ls = [record['h']['id'] for record in result]
        if len(ls) == 0:
            print("No phenotype")
            return [1]
        return ls

def upload_variants(patient_id, variants_df):
    with driver.session() as session:
        for idx, row in variants_df.iterrows():
            session.write_transaction(insert_variant, row.to_dict(), patient_id)






"""
driver = GraphDatabase.driver(uri, auth=(username, password))
geneSymbol = "ABCAA4"

with driver.session() as session:
    # Insert a Gene node with the given geneSymbol if it does not exist
    # so, check first if the gene exists
    result = session.run(
        "MATCH (g:Gene {geneSymbol: $geneSymbol}) RETURN g", {"geneSymbol": geneSymbol}
    )
    gene = result.single()

    if gene is None:
        # Gene does not exist, so create it
        session.run(
            "CREATE (g:Gene {geneSymbol: $geneSymbol}) RETURN g", {"geneSymbol": geneSymbol}
        )
    else:
        print("Gene already exists, skipping creation")
"""