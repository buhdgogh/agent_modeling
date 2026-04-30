import os
os.sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/..")
from ..utils import common

class population_density_api:
    def __init__(self):
        # load WorldPop dataset.
        self.dataset = common.ee.ImageCollection("WorldPop/GP/100m/pop").mosaic()

    def get_population_density(self, min_lon, min_lat, max_lon, max_lat, scale=100):
        # locate region
        roi = common.ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

        # clip specific region
        clipped_image = self.dataset.clip(roi)

        # calculate population of specific region
        population_total = clipped_image.reduceRegion(
            reducer=common.ee.Reducer.sum(),
            geometry=roi,
            scale=scale,
            maxPixels=1e8,
            bestEffort=True
        ).get("population").getInfo()

        # calculate region area (unit: m2)
        area_m2 = roi.area().getInfo()

        # convert area unit to km2 (1 km2 = 1,000,000 m2)
        area_km2 = max(1e-6, area_m2 / 1e6)

        # calculate population density (people / km2)
        population_density = round(population_total / area_km2, 2)
        population_density = f"{population_density} people/km^2"
        return population_density

if __name__ == "__main__":
    min_lon = 108.00#E
    max_lon = 109.00#E
    min_lat = 19.33#N
    max_lat = 20.00#N
    population_density_api = population_density_api()
    print(population_density_api.get_population_density(min_lon, min_lat, max_lon, max_lat))
