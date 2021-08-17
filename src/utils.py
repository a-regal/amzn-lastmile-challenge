import numpy as np
import torch

def distance_matrix(route_id, routes, travel_times):
    mat = np.zeros((len(routes[route_id]['stops']), len(routes[route_id]['stops'])))

    for i, stop in enumerate(travel_times[route_id]):
        mat[i, :] = list(travel_times[route_id][stop].values())

    return torch.tensor(mat, dtype=torch.float32)

def get_start_node(route_id, routes):
    start_node = -1
    for i, idx in enumerate(routes[route_id]['stops']):
        if routes[route_id]['stops'][idx]['type'] != 'Dropoff':
            start_node = i

    return start_node

def get_time_windows(route_id, packages):
    time_windows = []
    volumes = []
    service_times = []

    for stop in packages[route_id]:
        s, e = 0, 86400
        total_service_time = 0
        total_volume = 0
        for package in packages[route_id][stop]:
            start_time = packages[route_id][stop][package]['time_window']['start_time_utc']
            end_time = packages[route_id][stop][package]['time_window']['end_time_utc']

            if type(start_time) == str:
                s, e = start_time[-8:], end_time[-8:]
                s = int(s[:2])*60 + int(s[3:5])
                e = int(e[:2])*60 + int(e[3:5])
                total_service_time += packages[route_id][stop][package]['planned_service_time_seconds']
                total_volume += np.prod(list(packages[route_id][stop][package]['dimensions'].values()))
        
        time_windows.append((s,e))
        volumes.append(total_volume)
        service_times.append(total_service_time)

    return torch.tensor(time_windows, dtype=torch.float32), torch.tensor(volumes, dtype=torch.float32), torch.tensor(service_times, dtype=torch.float32)

def get_sequence(route_id, sequences):
        actual = sequences[route_id]['actual']
        d = dict(zip(range(len(actual)), actual.values()))
        a = list(dict(sorted(d.items(), key=lambda item: item[1])).keys())
        return a
