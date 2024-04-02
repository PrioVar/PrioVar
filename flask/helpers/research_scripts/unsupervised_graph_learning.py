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
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))  # This layer has been removed to output a single value per edge
        return x.mean(dim=0)  # Global mean pooling to output a single vector for all edges


# Define your edge_index and edge_weight tensors
edge_index = torch.load(path.join('../../data', 'edge_index.pt')).long().cuda()
edge_weight = torch.load(path.join('../../data', 'edge_weight.pt')).float().cuda()

# Define model parameters
num_nodes = edge_index.max().item() + 1
input_dim = 2048
hidden_dim = 64
embedding_dim = 32
num_epochs = 100

# Initialize the model
model = GraphAutoEncoder(input_dim, hidden_dim, embedding_dim, num_nodes).cuda()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
accumulation_steps = 10  # Accumulate gradients over 10 mini-batches
batch_size = 2048
number_of_edges = edge_index.size(1)
for epoch in range(num_epochs):
    optimizer.zero_grad()
    total_loss = 0
    for i in range(0, number_of_edges, batch_size):
        if i % 100 == 0:
            print(f'Processing batch {int(i / batch_size)}')
        optimizer.zero_grad()
        end_idx = min(i + batch_size, number_of_edges)
        edge_index_batch = edge_index[:, i:end_idx]
        edge_weight_batch = edge_weight[i:end_idx]

        recon_x, z = model(edge_index_batch)
        loss = criterion(recon_x, edge_weight_batch)  # Reconstruction loss using edge_weight
        loss.backward()

        total_loss += loss.item()

        if (i + 1) % accumulation_steps == 0 or i + batch_size >= number_of_edges:
            optimizer.step()
            optimizer.zero_grad()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, total_loss))

# After training, you can use the encoder to obtain node embeddings
with torch.no_grad():  # No need to compute gradients for this operation
    node_embeddings = model.embedding.detach().cpu().numpy()
