from flask import Flask, request, jsonify
from flask_cors import CORS

from helpers.hpo import read_hpo_from_json, process_nodes, process_edges, save_nodes, save_edges
from helpers.clinvar import read_clinvar, save_clinvar
from helpers.hpo_annotations import initiate_disease_database, initiate_gene_database
from helpers.annotation import annotate_variants

app = Flask(__name__)
CORS(app)


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
    annotated_df = annotate_variants(file)
    # return the annotated variants as a dataframe
    return annotated_df.to_json()
    
if __name__ == '__main__':
    app.run(debug=True, port=5001)

