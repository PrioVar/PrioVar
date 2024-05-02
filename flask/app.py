from flask import Flask, request
from flask_cors import CORS

from helpers.hpo import read_hpo_from_json, process_nodes, process_edges, save_nodes, save_edges
from helpers.clinvar import read_clinvar, save_clinvar
from helpers.hpo_annotations import initiate_disease_database, initiate_gene_database
from helpers.annotation import annotate_variants, get_all_annotated_variants
from helpers.knowledge_graph import get_answer
from helpers.ClinicalResearchAssistant import analyze
from helpers.file_decode import read_file_content_and_return_df

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

    df = read_file_content_and_return_df(vcf_id)

    return "hey"

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


if __name__ == '__main__':
    app.run(debug=True, port=5001)

