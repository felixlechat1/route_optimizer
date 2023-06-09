import pandas as pd
import requests # to get the distances from the API
import json # to read the API response
import mlrose # for travelling salesman problem
from geopy.geocoders import Nominatim


def get_distance(point1: dict, point2: dict) -> tuple:
    """Gets distance between two points en route using http://project-osrm.org/docs/v5.10.0/api/#route-service"""

    url = f"""http://router.project-osrm.org/route/v1/driving/{point1["lon"]},{point1["lat"]};{point2["lon"]},{point2["lat"]}?overview=false&alternatives=false"""
    
    r = requests.get(url)

    # get the distance from the returned values
    route = json.loads(r.content)["routes"][0]
    return route["distance"], route["duration"]


def compute_tsp(dataframe, name):
    """Generates the graph for every combination of address and their distances
    and computes  the TSP problem of the obtained graph"""
    df = dataframe
    dist_array = []

    # compute distances between all adresses
    for i, r in df.iterrows():
        point1 = {"lat": r["lat"], "lon": r["lon"]}
        for j, o in df[df.index != i].iterrows():
            point2 = {"lat": o["lat"], "lon": o["lon"]}
            dist, duration = get_distance(point1, point2)
            dist_array.append((i, j, dist))
        print("Computing distances for address " + str(i+1))
    print("Graph built\n")
    print("Starting TSP solving")

    # genetic algorithm parameters to solve TSP problem over all distances and addresses
    length = df.shape[0]
    fitness_dists = mlrose.TravellingSales(distances=dist_array)
    problem_fit = mlrose.TSPOpt(length=length, fitness_fn=fitness_dists)

    # compute genetic algorithm
    best_state, best_fitness = mlrose.genetic_alg(problem_fit, pop_size=200, mutation_prob=0.2, max_attempts=3000, random_state=2)

    print(f"The best state found is: {best_state}, taking {best_fitness/1000} km")

    # reorder list
    orders = {city: order for order, city in enumerate(best_state)}
    df["order"] = df.index.map(orders)
    df = df.sort_values(by="order")
    print(df)

    # remove unwanted columns and export ordered list
    df = df.drop(columns=['Unnamed: 0', 'lat', 'lon', 'order'])
    df.to_csv(name+'_out.csv', index=False)

def compute_tspV2(dataframe, name):
    df = dataframe
    points=""
    for i, r in df.iterrows():
        points+=str(r["lon"])
        points+=','
        points+=str(r["lat"])
        points+=';'
    points = points[:-1]
    url = f"""http://router.project-osrm.org/trip/v1/driving/{points}?overview=false"""
    print(url)
    r = requests.get(url)
    res = json.loads(r.content)
    print(res['waypoints'])

    order = []
    for waypoint in res:
        print(waypoint)
        order.append(waypoint['waypoint_index'])
def load_coord(dataframe):
    """Loads the coordinates to each address and inserts it in the dataframe"""
    geolocator = Nominatim(user_agent="route_optimizer")
    for i in range(dataframe['addr'].shape[0]):
        location = geolocator.geocode(dataframe['addr'][i])
        print(location)


if __name__ == '__main__':
    name = 'CDN'
    dataframe = pd.read_csv(name + ".csv")
    compute_tsp(dataframe, name)

    #load_coord(dataframe)


