import os
os.sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/..")
import json
from ..utils import api, prompt, vision, common
from ..tool_pool import landcover_type_api
from ..tool_pool import population_density_api

class geographer_agent:
    def __init__(self):
        self.landcover_type_api = landcover_type_api()
        self.population_density_api = population_density_api()

    def get_knowledge(self, min_lon, min_lat, max_lon, max_lat):
        landcover_distribution = self.landcover_type_api.get_landcover_distribution(min_lon, min_lat, max_lon, max_lat)
        population_density = self.population_density_api.get_population_density(min_lon, min_lat, max_lon, max_lat)
        geographical_data = {
            "landcover_distribution": landcover_distribution, 
            "population_density": population_density,
        }
        return geographical_data

if __name__ == "__main__":
    min_lon = 108.00#E
    max_lon = 109.00#E
    min_lat = 19.33#N
    max_lat = 20.00#N
    geographer_agent = geographer_agent()
    print(geographer_agent.get_knowledge(min_lon, min_lat, max_lon, max_lat))
