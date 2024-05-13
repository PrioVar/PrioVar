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
from helpers.api_functions import (
    api_start_analysis, api_get_output,
    set_vcf_file_details_for_patient, set_vcf_file_details,
    upload_variants, get_patient_phenotypes,
    save_annotated_vcf_file
)
from helpers.ml_model import get_mock_results
import time


app = Flask(__name__)
CORS(app)


@app.route('/ai-help', methods=['POST'])
def ai_support():
    data = request.get_json()
    return analyze(data)


# write an endpoint that takes a question as an input and queries the knowledge graph
# to return an answer
@app.route('/search-graph', methods=['POST'])
def search_graph():
    # get the question, which is a parameter in the request
    data = request.get_json()
    question = data.get('question')
    medical_center_id = data.get('healthCenterId')

    if question is None:
        return "No question provided"

    # query the knowledge graph
    answer = get_answer(question, medical_center_id)

    # return the answer
    return answer


@app.route('/analysis', methods=['POST'])
def start_analysis():
    data = request.get_json()
    vcf_id = data.get('vcfId')

    if vcf_id is None:
        return "No VCF ID provided"

    #df = read_file_content_and_return_df(vcf_id)

    set_vcf_file_details(vcf_id, "abfxh8559" ,"ANALYSIS_IN_PROGRESS")

    return "hey"


# return mock analysis results, for testing purposes, similar to above
@app.route('/analysis-mock', methods=['POST'])
def start_analysis_mock():

    # read patient id from the request
    data = request.get_json()
    patient_id = data.get('patientId')

    set_vcf_file_details_for_patient(patient_id, "ANALYSIS_IN_PROGRESS")

    #hpo_list = data.get('hpoList')
    hpo_list = get_patient_phenotypes(patient_id)
    #print("Hpolist: ", hpo_list)

    df = get_mock_results(16, hpo_list)

    # print df columns
    #print("Columns: ", df.columns)

    upload_variants(patient_id, df)

    # set the vcf file details for the patient
    set_vcf_file_details_for_patient(patient_id, "ANALYSIS_DONE")

    return df.to_json()


@app.route('/load-hpo', methods=['GET'])
def start_loading_data():
    graph_id, meta, nodes, edges, property_chain_axioms = read_hpo_from_json()

    items = process_nodes(nodes)
    save_nodes(items)
    edge_items = process_edges(edges)
    save_edges(edge_items)

    return "Data loading finished"


@app.route('/load-clinvar', methods=['GET'])
def get_clinvar():
    df = read_clinvar()
    save_clinvar(df)

    return "Variants from ClinVar successfully loaded"


@app.route('/load-diseases', methods=['GET'])
def get_diseases():
    initiate_disease_database()

    return "Diseases successfully loaded"


@app.route('/load-genes', methods=['GET'])
def get_genes():
    initiate_gene_database()

    return "Genes successfully loaded"


# write an endpoint that takes a VCF file as input and returns annotated variants
# as a dataframe
@app.route('/annotate-variants', methods=['POST'])
def get_annotated_variants():
    # get the file from the request
    file = request.files['file']
    # annotate the variants
    annotated_data = annotate_variants(file)
    # return the annotated variants as a dataframe
    return annotated_data


@app.route('/get-annotated-variants', methods=['GET'])
def get_annotated_variants_of_patient():
    return get_all_annotated_variants()


@app.route("/endpoint-test", methods=["GET"])
def test_endpoint():
    """
    FULL PIPELINE TEST
    :return:
    """
    # Prepare the file to be uploaded with an explicit filename
    # Open the file in binary mode
    with open('data/tinyy.vcf', 'rb') as file:
        file_content = file.read()

    # now, send a request to lidyagenomics.com/libra/api/v1/vcf/cs492upload
    # keep the same headers as above
    # also, request.data["file"] should be the file to be uploaded, which is data/tinyy.vcf
    # Request should include a Content - Disposition header with a filename parameter
    response = requests.post("http://lidyagenomics.com/libra/api/v1/vcf/cs492upload", headers={
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
        "Content-Disposition": "attachment; filename=tinyy.vcf",
        "Content-Type": "application/octet-stream"  # For binary data
    }, data=file_content)

    print(response.json())
    vcf_id = response.json()['vcf_id']

    # start an analysis with the uploaded file
    start_response = api_start_analysis(vcf_id)
    print(start_response)

    # in every 15 seconds, check the status of the analysis, if 200,
    # save the file to data/annotated_vcfs folder
    # if not, continue checking
    while True:
        status_response = api_get_output(vcf_id)
        if status_response.status_code == 200:
            # Assuming the response contains the file content to save
            file_path = f"data/annotated_vcfs/{vcf_id}.vcf"
            with open(file_path, 'wb') as file:
                file.write(status_response.content)
            print(f"File successfully saved at {file_path}")
            break  # Exit the loop since the file has been saved
        else:
            # If the status is not 200, print the status and wait 15 seconds before checking again
            print(f"Status code: {status_response.status_code}. Checking again in 15 seconds.")
            time.sleep(15)

    # Temporary filenames

    return "Endpoint test successful"


# endpoint test2
@app.route("/endpoint-test2", methods=["GET"])
def test_endpoint2():
    vcf_sample = {'vcf_id': 'ea129c59-3382-41ef-990e-e9746ff958d1'}
    vcf_sample2 = {'vcf_id': '59bb252b-45e9-4086-81a4-a4e3e11c6935'}

    #start an analysis with the uploaded file
    response2 = api_start_analysis(vcf_sample2['vcf_id'])
    print(response2)
    #print(response2.json())
    # save the response to a file

    return "Endpoint2 test successful"


@app.route("/endpoint-test3", methods=["GET"])
def test_endpoint3():
    vcf_sample = {'vcf_id': 'ea129c59-3382-41ef-990e-e9746ff958d1'}
    vcf_sample2 = {'vcf_id': '59bb252b-45e9-4086-81a4-a4e3e11c6935'}

    # Pick one
    vcf_sample_id = vcf_sample2['vcf_id']

    # get the output of the analysis
    response3 = api_get_output(vcf_sample_id)
    print(response3)

    save_annotated_vcf_file(response3.content, vcf_sample_id)

    return "Endpoint3 test successful"


if __name__ == '__main__':
    app.run(debug=True, port=5001)

