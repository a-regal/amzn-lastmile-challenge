def sort_by_key(stops, sort_by):
    """
    Takes in the `prediction_routes[route_id]['stops']` dictionary
    Returns a dictionary of the stops with their sorted order always placing the depot first

    EG:

    Input:
    ```
    stops={
      "Depot": {
        "lat": 42.139891,
        "lng": -71.494346,
        "type": "depot",
        "zone_id": null
      },
      "StopID_001": {
        "lat": 43.139891,
        "lng": -71.494346,
        "type": "delivery",
        "zone_id": "A-2.2A"
      },
      "StopID_002": {
        "lat": 42.139891,
        "lng": -71.494346,
        "type": "delivery",
        "zone_id": "P-13.1B"
      }
    }

    print (sort_by_key(stops, 'lat'))
    ```

    Output:
    ```
    {
        "Depot":1,
        "StopID_001":3,
        "StopID_002":2
    }
    ```

    """
    # Serialize keys as id into each dictionary value and make the dict a list
    stops_list=[{**value, **{'id':key}} for key, value in stops.items()]

    # Sort the stops list by the key specified when calling the sort_by_key func
    ordered_stop_list=sorted(stops_list, key=lambda x: x[sort_by])

    # Keep only sorted list of ids
    ordered_stop_list_ids=[i['id'] for i in ordered_stop_list]

    # Serialize back to dictionary format with output order as the values
    return {i:ordered_stop_list_ids.index(i) for i in ordered_stop_list_ids}

def propose_all_routes(prediction_routes, sort_by):
    """
    Applies `sort_by_key` to each route's set of stops and returns them in a dictionary under `output[route_id]['proposed']`

    EG:

    Input:
    ```
    prediction_routes = {
      "RouteID_001": {
        ...
        "stops": {
          "Depot": {
            "lat": 42.139891,
            "lng": -71.494346,
            "type": "depot",
            "zone_id": null
          },
          ...
        }
      },
      ...
    }

    print(propose_all_routes(prediction_routes, 'lat'))
    ```

    Output:
    ```
    {
      "RouteID_001": {
        "proposed": {
          "Depot": 0,
          "StopID_001": 1,
          "StopID_002": 2
        }
      },
      ...
    }
    ```
    """
    return {key:{'proposed':sort_by_key(stops=value['stops'], sort_by=sort_by)} for key, value in prediction_routes.items()}


# print('\nApplying answer with real model...')
# sort_by=model_build_out.get("sort_by")
# print('Sorting data by the key: {}'.format(sort_by))
# output=propose_all_routes(prediction_routes=prediction_routes, sort_by=sort_by)
# print('Data sorted!')
