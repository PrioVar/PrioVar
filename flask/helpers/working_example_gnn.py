import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn.functional as F

class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        return self.conv2(x, edge_index, edge_weight)

# Constants
latent_dimension = 16  # Example latent dimension
num_nodes = 10  # Example number of nodes
edge_indexes = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  # Example edge indexes
edge_weights = torch.tensor([0.8, 0.5, 0.9], dtype=torch.float)  # Example edge weights

# Initialize node features randomly
node_features = torch.randn(num_nodes, latent_dimension)

# Create a GNN model
model = GNN(latent_dimension, latent_dimension)

# Create a data object
data = Data(x=node_features, edge_index=edge_indexes, edge_attr=edge_weights)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# TODO: replace here with other type of distances/similiarities
def cosine_similarity_normalized(u, v):
    # Cosine similarity normalized to be between 0 and 1
    cos_sim = F.cosine_similarity(u, v, dim=-1)
    return (cos_sim + 1) / 2

def train():
    model.train()
    optimizer.zero_grad()
    z = model(data.x, data.edge_index, data.edge_attr)

    # Get embeddings of source and target nodes
    source_embeddings = z[data.edge_index[0]]
    target_embeddings = z[data.edge_index[1]]

    # Calculate normalized cosine similarity
    cos_sim_normalized = cosine_similarity_normalized(source_embeddings, target_embeddings)

    # Calculate the Mean Squared Error loss
    loss = F.mse_loss(cos_sim_normalized, data.edge_attr, reduction='none').mean()

    loss.backward()
    optimizer.step()
    return loss

# Training loop
for epoch in range(200):
    loss = train()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')