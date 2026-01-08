import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCN, self).__init__()

        # Graph convolution layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        # Final classifier
        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        # 1st GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # 2nd GCN layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Graph-level pooling
        x = global_mean_pool(x, batch)

        # Classification
        x = self.lin(x)

        return x

