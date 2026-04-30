import geopandas as gpd
from shapely.geometry import box
import os

class active_fault_db:
    def __init__(self, db_path=os.path.join(os.path.dirname(__file__), "..", "dependencies", "knowledge", "gem_active_faults_harmonized.geojson")):
        self.fault_db = gpd.read_file(db_path)

    def get_active_faults(self, min_lon, min_lat, max_lon, max_lat):
        # Create a bounding box from the latitude and longitude ranges
        bbox = box(min_lon, min_lat, max_lon, max_lat)
        # Filter the LineStrings that intersect with the bounding box
        intersecting_lines = self.fault_db[self.fault_db.intersects(bbox)]

        # Check if there are any intersecting lines
        if intersecting_lines.empty:
            #print("No active fault lines intersect with the given range.")
            active_fault_lines = "No active fault within the given range."
        else:
            # Compute the intersecting geometries
            intersecting_lines = intersecting_lines.copy()  # Copy to avoid warnings
            intersecting_lines["intersection"] = intersecting_lines.geometry.intersection(bbox)

            # Estimate an appropriate projected CRS (e.g., UTM)
            utm_crs = intersecting_lines.estimate_utm_crs()

            # Set the intersecting geometries as the active geometry and project to the UTM CRS
            intersecting_lines = intersecting_lines.set_geometry("intersection")
            intersecting_lines = intersecting_lines.to_crs(utm_crs)

            # Calculate the length of the intersecting parts (in meters)
            intersecting_lines["length_m"] = intersecting_lines.length
            # Convert the lengths to kilometers
            intersecting_lines["length_in_kilometers"] = round(intersecting_lines["length_m"] / 1000.0, 2)

            selected_columns = ["slip_type", "name", "catalog_name", "length_in_kilometers", "dip_dir", "average_dip", "average_rake", "lower_seis_depth", "upper_seis_depth"]
            active_fault_lines = intersecting_lines[selected_columns].to_dict(orient="records")

        return active_fault_lines

if __name__ == "__main__":
    min_lon = 99.00  # E
    max_lon = 100.00  # E
    min_lat = 27.33  # N
    max_lat = 28.00  # N
    active_fault_db = active_fault_db()
    print(active_fault_db.get_active_faults(min_lon, min_lat, max_lon, max_lat))
