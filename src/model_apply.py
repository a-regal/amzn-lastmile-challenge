from os import path
import sys, json, time
import torch
import numpy as np
import torch.nn.functional as F
from model import GraphNet
from tqdm.auto import tqdm
from utils import *
from torch_geometric.data import Data

# Get Directory
BASE_DIR = path.dirname(path.dirname(path.abspath(__file__)))

# Read input data
print('Reading Weights into network')

# Model Build output
model_path=path.join(BASE_DIR, 'data/model_build_outputs/gcn_state_dict.pt')
print(model_path)
gn = GraphNet(4)
gn.load_state_dict(torch.load(model_path))

print('Reading new dataset')
test_routes_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_route_data.json')
test_travel_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_travel_times.json')
test_pacakge_path = path.join(BASE_DIR, 'data/model_apply_inputs/new_package_data.json')

print('Reading route data')
with open(test_routes_path, 'r', newline='') as f:
    routes = json.load(f)

print('Reading travel time data')
with open(test_travel_path, 'r', newline='') as f:
    travel_times = json.load(f)

print('Reading package data')
with open(test_pacakge_path, 'r', newline='') as f:
    packages = json.load(f)

route_ids = [key for key in routes.keys()]

def sequencer(route_id, route_keys, model):
    sequence = []
    matrix = distance_matrix(route_id, routes, travel_times)
    tw, vl, st = get_time_windows(route_id, packages)
    stop = get_start_node(route_id, routes)

    available = set(range(matrix.shape[0]))
    visited = set([])
    sequence.append(stop)

    indices = list(set(available) - set(visited))
    node_features = torch.cat([tw[:,0].view(-1,1), tw[:,1].view(-1,1), vl.view(-1,1), st.view(-1,1)], axis=1)

    edge_attrs_ = matrix[stop, indices]
    x_ = node_features[indices, :]
    edge_index_ = torch.tensor([[indices[stop] for i in range(len(indices))],
                        [i for i in range(len(indices))]], dtype=torch.long)


    for step in range(matrix.shape[0]):
        if len(visited) == matrix.shape[ 0] -1:
            sequence.append(list(available - visited)[0])
            break
        else:

            graph = Data(x=x_, edge_index=edge_index_, edge_attr=edge_attrs_)
            out = model(graph)
            selected_node = torch.argmax(F.softmax(out, dim=1)).item()
            selected_stop = indices[selected_node]

            sequence.append(selected_stop)
            visited.add(selected_stop)
            stop = selected_stop

            indices = list(available - visited)
            edge_attrs_ = matrix[stop, indices]
            x_ = node_features[indices, :]

            targets = list(range(len(indices)))
            if selected_node == len(indices):
                edge_index_ = torch.tensor([[targets[-1] for i in targets],
                                            [i for i in targets]], dtype=torch.long)
            else:
                edge_index_ = torch.tensor([[targets[selected_node] for i in targets],
                                            [i for i in targets]], dtype=torch.long)

    sequence_dict = {route_keys[idx]: i for i, idx in enumerate(sequence)}

    return sequence_dict

print('Start sequencing')
proposed_routes = {route_id: {'proposed':sequencer(route_id, list(routes[route_id]['stops'].keys()), gn)} for route_id in tqdm(route_ids)}

# Write output data
output_path = path.join(BASE_DIR, 'data/model_apply_outputs/proposed_sequences.json')

with open(output_path, 'w') as out_file:
    json.dump(proposed_routes, out_file)
    print("Success: The '{}' file has been saved".format(output_path))

print('Done!')
