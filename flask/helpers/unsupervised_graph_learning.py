import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from os import path


class GraphAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_nodes):
        super(GraphAutoEncoder, self).__init__()
        self.embedding = nn.Parameter(torch.randn(num_nodes, embedding_dim))  # Learnable node embeddings
        self.encoder = GCNEncoder(hidden_dim, embedding_dim)
        self.decoder = GCNDecoder(embedding_dim, hidden_dim, input_dim)

    def forward(self, edge_index):
        z = self.encoder(self.embedding, edge_index)
        recon_x = self.decoder(z, edge_index)
        return recon_x, z


class GCNEncoder(nn.Module):
    def __init__(self, hidden_dim, embedding_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(embedding_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x


class GCNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNDecoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # Adjust the input dimensionality of x for compatibility with convolutional layers
        x = F.relu(self.conv1(x, edge_index))
        recon_x = self.conv2(x, edge_index)
        return recon_x


# Define your edge_index and edge_weight tensors
edge_index = torch.load(path.join('../data', 'edge_index.pt')).long()
edge_weight = torch.load(path.join('../data', 'edge_weight.pt')).float()

# Define model parameters
num_nodes = edge_index.max().item() + 1
input_dim = edge_weight.size(0)  # Assuming the input dimension is equal to the number of nodes
hidden_dim = 64
embedding_dim = 32
num_epochs = 100

# Initialize the model
model = GraphAutoEncoder(input_dim, hidden_dim, embedding_dim, num_nodes)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    recon_x, z = model(edge_index)
    loss = criterion(recon_x, edge_weight)  # Reconstruction loss using edge_weight
    loss.backward()
    optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# After training, you can use the encoder to obtain node embeddings
node_embeddings = model.embedding
