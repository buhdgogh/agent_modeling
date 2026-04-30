import os
os.sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/..")
from ..utils import common

class landcover_type_api:
    def __init__(self):
        # load ESA WorldCover image
        self.dataset = common.ee.ImageCollection("ESA/WorldCover/v200").mosaic()

        # define the landcover type
        self.landcover_class_names = {
            10: "Trees",
            20: "Shrubland",
            30: "Grassland",
            40: "Cropland",
            50: "Built-up",
            60: "Bare / Sparse Vegetation",
            70: "Snow and Ice",
            80: "Permanent Water Bodies",
            90: "Herbaceous Wetland",
            95: "Mangroves",
            100: "Moss and Lichen"
        }

    def get_landcover_distribution(self, min_lon, min_lat, max_lon, max_lat, scale=100):
        # locate region
        roi = common.ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

        # clip specific region
        clipped_image = self.dataset.clip(roi)

        # calculate distribution of each landcover type
        reducer = common.ee.Reducer.frequencyHistogram()
        pixel_counts = clipped_image.reduceRegion(
            reducer=reducer,
            geometry=roi,
            scale=scale,
            maxPixels=1e8,
            bestEffort=True
        )

        # get statistic result
        pixel_histogram = pixel_counts.get("Map").getInfo()

        # calculate ratios of each landcover type
        total_pixels = max(1, sum(pixel_histogram.values()))
        pixel_percentages = {self.landcover_class_names[int(k)]: round((v / total_pixels) * 100.0, 3) for k, v in pixel_histogram.items()}
        landcover_distribution = pixel_percentages
        return landcover_distribution

if __name__ == "__main__":
    min_lon = 108.00#E
    max_lon = 109.00#E
    min_lat = 19.33#N
    max_lat = 20.00#N
    landcover_type_api = landcover_type_api()
    print(landcover_type_api.get_landcover_distribution(min_lon, min_lat, max_lon, max_lat))
