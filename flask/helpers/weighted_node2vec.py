import networkx as nx
import torch

from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
from gensim.models import Word2Vec


def read_tensor_data(edge_index_path, edge_weight_path):
    edge_index = torch.load(edge_index_path)
    edge_weight = torch.load(edge_weight_path)
    G = nx.Graph()
    edge_set = set()
    count = 0
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        weight = edge_weight[i].item()

        # if weight is equal to 1, rescale it to 5 x 10^-5
        if weight == 1:
            weight = 5 * 10 ** -5

        # Check if the edge already exists in the set
        if (src, dst) in edge_set:
            # Update the weight of the existing edge with the larger weight value
            G[src][dst]["weight"] = max(G[src][dst]["weight"], weight)
            count += 1
            if count % 1000 == 0:
                print(f"source: {src}, destination: {dst}, weight: {weight}")
        else:
            # Add the edge to the graph and the set
            G.add_edge(src, dst, weight=weight)
            edge_set.add((src, dst))
    return G


# read the tensors data/edge_index.pt and data/edge_weight.pt into a networkx graph
G = read_tensor_data("../data/edge_index.pt", "../data/edge_weight.pt")

# create a StellarGraph object from the networkx graph
G = StellarGraph.from_networkx(G)
rw = BiasedRandomWalk(G)

# PARAMETERS
WALK_LENGTH = 100  # maximum length of a random walk
n = 5  # number of random walks per root node
p = 0.5  # Defines (unormalised) probability, 1/p, of returning to source node
q = 2  # Defines (unormalised) probability, 1/q, for moving away from source node
vector_size = 256  # dimension of node embeddings

weighted_walks = rw.run(
    nodes=G.nodes(),  # root nodes
    length=WALK_LENGTH,  # maximum length of a random walk
    n=n, p=p, q=q,
    weighted=True,  # for weighted random walks
    seed=42,  # random seed fixed for reproducibility
)
print("Number of random walks: {}".format(len(weighted_walks)))

weighted_model = Word2Vec(
    weighted_walks, vector_size=vector_size, window=5, min_count=5, sg=1, workers=2, epochs=5
)

# Save all embeddings into a file manually
file_path = f"../data/node_embeddings_p{p}_q{q}_n{n}_v{vector_size}.txt"
with open(file_path, "w") as f:
    f.write(f"{len(weighted_model.wv)} {weighted_model.vector_size}\n")
    keys = weighted_model.wv.index_to_key
    vectors = weighted_model.wv.vectors
    for i in range(len(keys)):
        f.write(f"{keys[i]} {' '.join(map(str, vectors[i]))}\n")

# maybe increase n???
