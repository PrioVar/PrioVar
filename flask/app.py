from flask import Flask, request, jsonify
from flask_cors import CORS

from helpers.hpo import read_hpo_from_json, process_nodes, process_edges, save_nodes, save_edges

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


if __name__ == '__main__':
    app.run(debug=True, port=5001)

