from flask import Flask, request
from flask_cors import CORS

from helpers.hpo import read_hpo_from_json, process_nodes, process_edges, save_nodes, save_edges
from helpers.clinvar import read_clinvar, save_clinvar
from helpers.hpo_annotations import initiate_disease_database, initiate_gene_database
from helpers.annotation import annotate_variants, get_all_annotated_variants
from helpers.knowledge_graph import get_answer
from helpers.ClinicalResearchAssistant import analyze
from helpers.file_decode import read_file_content, read_file_content_and_return_df
from helpers.api_functions import (
    api_start_analysis, api_get_output,
    set_vcf_file_details_for_patient, set_vcf_file_details,
    upload_variants, get_patient_phenotypes,
    save_annotated_vcf_file, api_upload_vcf_file
)
from helpers.ml_model import get_real_results
import time
import pandas as pd

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


@app.route('/analysis-test', methods=['POST'])
def start_analysis_test():

    file_name = "1e84230c-b960-49ea-be08-1d28341a078a.vcf"

    df = get_real_results(file_name, [1])

    def safe_mode(series):
        try:
            # Attempt to return the most common value
            return series.mode().iloc[0]
        except IndexError:
            # If no mode found or empty, return NaN or another placeholder like None
            return "-"

    # Group by '#Uploaded_variation' and aggregate using the safe mode function
    result = df.groupby('#Uploaded_variation').agg(safe_mode)

    # save df as file
    result.to_csv("temp.csv")

    return "hey"


# return analysis results
@app.route('/analysis-mock', methods=['POST'])
def start_analysis_mock():

    # read patient id from the request
    data = request.get_json()
    patient_id = data.get('patientId')
    priovar_vcf_id = data.get('vcfId')

    # Set the vcf file details for the patient and get phenotypes
    set_vcf_file_details_for_patient(patient_id, "ANALYSIS_IN_PROGRESS")
    hpo_list = get_patient_phenotypes(patient_id)

    # get decoded file and upload
    file_content = read_file_content(priovar_vcf_id)
    vcf_id = api_upload_vcf_file(file_content)

    # TODO: YOU MAY BENEFIT FROM THE BELOW DATAFRAME TO GET CHROM, POS, etc.
    file_content_df = read_file_content_and_return_df(priovar_vcf_id)
    print("len of file_content_df: ", len(file_content_df))

    # start an analysis with the uploaded file
    start_response = api_start_analysis(vcf_id)
    print(start_response)

    # in every 15 seconds, check the status of the analysis, if 200,
    # save the file, if not, continue checking
    time.sleep(10)
    while True:
        status_response = api_get_output(vcf_id)
        if status_response.status_code == 200:
            # TODO: YOU MIGHT WANT TO RETURN THE FILE INSTEAD OF SAVING IT
            # TODO: BUT YOU MAY ALSO READ IT FROM THE SAVED FILE, UP TO YOU
            final_file_name = save_annotated_vcf_file(status_response.content, vcf_id)
            print(f"File successfully saved!")
            break  # Exit the loop since the file has been saved
        else:
            # If the status is not 200, print the status and wait 15 seconds before checking again
            print(f"Status code: {status_response.status_code}. Checking again in 15 seconds.")
            time.sleep(15)

    # TODO: CORRECT HERE
    def safe_mode(series):
        try:
            # Attempt to return the most common value
            return series.mode().iloc[0]
        except IndexError:
            # If no mode found or empty, return NaN or another placeholder like None
            return "-"

    # Group by '#Uploaded_variation' and aggregate using the safe mode function
    df = get_real_results(final_file_name, hpo_list)

    df["group_by_column"] = df["#Uploaded_variation"]

    df = df.groupby('group_by_column').agg(safe_mode)

    df.reset_index(drop=True, inplace=True)
    file_content_df.reset_index(drop=True, inplace=True)

    # concatenate the two dataframes
    df = pd.concat([df, file_content_df], axis=1)

    upload_variants(patient_id, df)

    # set the vcf file details for the patient
    set_vcf_file_details_for_patient(patient_id, "ANALYSIS_DONE", vcf_id)

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
    FULL PIPELINE
    :return:
    """
    # Prepare the file to be uploaded with an explicit filename
    # Open the file in binary mode
    with open('data/tinyy.vcf', 'rb') as file:
        file_content = file.read()

    vcf_id = api_upload_vcf_file(file_content)

    # start an analysis with the uploaded file
    start_response = api_start_analysis(vcf_id)
    print(start_response)

    # in every 15 seconds, check the status of the analysis, if 200,
    # save the file to data/annotated_vcfs folder
    # if not, continue checking
    time.sleep(10)
    while True:
        status_response = api_get_output(vcf_id)
        if status_response.status_code == 200:
            # Assuming the response contains the file content to save
            save_annotated_vcf_file(status_response.content, vcf_id)
            print(f"File successfully saved!")
            break  # Exit the loop since the file has been saved
        else:
            # If the status is not 200, print the status and wait 15 seconds before checking again
            print(f"Status code: {status_response.status_code}. Checking again in 15 seconds.")
            time.sleep(15)

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

