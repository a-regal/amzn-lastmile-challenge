import torch.nn as nn
from torch_geometric.nn import GCNConv

class GraphNet(nn.Module):
    def __init__(self, num_features):
        super(GraphNet, self).__init__()
        self.conv1 = GCNConv(num_features, 1)
        self.mish = nn.Mish()

    def forward(self, data):
        x, edge_index, attrs = data.x, data.edge_index, data.edge_attr
        x = self.mish(self.conv1(x, edge_index, edge_weight=attrs))
        return x.view(1,-1)
