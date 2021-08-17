from os import path
import sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from utils import *
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from tqdm.auto import tqdm

EPOCHS = 100

class GraphNet(nn.Module):
    def __init__(self, num_features):
        super(GraphNet, self).__init__()
        self.conv1 = GCNConv(num_features, 1)
        self.mish = nn.Mish()

    def forward(self, data):
        x, edge_index, attrs = data.x, data.edge_index, data.edge_attr
        x = self.mish(self.conv1(x, edge_index, edge_weight=attrs))
        return x.view(1,-1)

# Get Directory
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))

# Read input data
print('Reading Input Data')
training_routes_path = path.join(BASE_DIR, 'data/model_build_inputs/route_data.json')
training_travel_path = path.join(BASE_DIR, 'data/model_build_inputs/travel_times.json')
training_pacakge_path = path.join(BASE_DIR, 'data/model_build_inputs/package_data.json')
training_sequence_path = path.join(BASE_DIR, 'data/model_build_inputs/actual_sequences.json')

print('Reading Route Data')
with open(training_routes_path, 'r', newline='') as f:
    routes = json.load(f)

print('Reading Travel Time Data')
with open(training_travel_path, 'r', newline='') as f:
    travel_times = json.load(f)

print('Reading Package Data')
with open(training_pacakge_path, 'r', newline='') as f:
    packages = json.load(f)

print('Reading Sequence Data')
with open(training_sequence_path, 'r', newline='') as f:
    sequences = json.load(f)

route_ids = [key for key in routes.keys() if routes[key]['route_score'] == 'High']

def make_dataset(route_ids):
    graphs = []
    target = []

    for route_id in tqdm(route_ids):
        matrix = distance_matrix(route_id, routes, travel_times)
        tw, vl, st = get_time_windows(route_id, packages)
        seq = get_sequence(route_id, sequences)
        current = get_start_node(route_id, routes)

        available = range(len(seq))
        visited = set([])

        indices = list(set(available) - visited)

        node_features = torch.cat([tw[:,0].view(-1,1), tw[:,1].view(-1,1), vl.view(-1,1), st.view(-1,1)], axis=1)

        for i, stop in enumerate(seq[:-1]):
            edge_attrs_ = matrix[stop, indices]
            edge_index_ = torch.tensor([[indices.index(stop) for i in range(len(indices))],\
                                        [i for i in range(len(indices))]], dtype=torch.long)
            x_ = node_features[indices, :]

            graphs.append(Data(x=x_, edge_index=edge_index_, edge_attr=edge_attrs_))
            target.append(indices.index(seq[i+1]))
            visited.add(stop)

            indices = list(set(available) - visited)
            current = stop

    return graphs, torch.tensor(target)

print('Create dataset')
x, y = make_dataset(route_ids)

gn = GraphNet(4)
optimizer = optim.Adam(gn.parameters(), 3e-4)
criterion = nn.CrossEntropyLoss()

print('Begin training')
for epoch in tqdm(range(EPOCHS)):
    for graph, target in zip(x,y):
        optimizer.zero_grad()
        out = gn(graph)
        loss = criterion(out, target.view(1))
        loss.backward()
        optimizer.step()

print('Save model')
model_path=path.join(BASE_DIR, 'data/model_build_outputs/gcn_state_dict.pt')
torch.save(gn.state_dict(), model_path)
