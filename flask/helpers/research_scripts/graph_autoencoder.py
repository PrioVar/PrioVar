import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the Graph Autoencoder model
class GraphAutoencoder(nn.Module):
    def __init__(self, num_nodes, latent_dim):
        super(GraphAutoencoder, self).__init__()
        self.encoder = nn.Linear(num_nodes, latent_dim)
        self.decoder = nn.Linear(latent_dim, num_nodes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        reconstructed = self.sigmoid(decoded)
        return reconstructed

# Generate a random adjacency matrix for illustration purposes
num_nodes = 50
adjacency_matrix = torch.tensor(np.random.randint(2, size=(num_nodes, num_nodes)), dtype=torch.float32)
print(adjacency_matrix)

# Create and train the Graph Autoencoder
latent_dim = 20
model = GraphAutoencoder(num_nodes, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(adjacency_matrix)
    loss = criterion(outputs, adjacency_matrix)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Obtain node embeddings using the trained encoder
node_embeddings = model.encoder(adjacency_matrix).detach().numpy()

# Print the learned embeddings for the first node
print("Node Embeddings for the First Node:")
print(node_embeddings[0])
