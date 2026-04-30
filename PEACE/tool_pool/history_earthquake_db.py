import pandas as pd
import os

class history_earthquake_db:
    def __init__(self, db_path=os.path.join(os.path.dirname(__file__), "..", "dependencies", "knowledge", "earthquake_1970_4.5mag.csv")):
        self.earthquake_db = pd.read_csv(db_path)

    def get_earthquake_history(self, min_lon, min_lat, max_lon, max_lat):
        margin = 0.05
        lat_min = min_lat - margin
        lat_max = max_lat + margin
        lon_min = min_lon - margin
        lon_max = max_lon + margin
        filtered_df = self.earthquake_db[(self.earthquake_db["latitude"]  >= lat_min) & (self.earthquake_db["latitude"]  <= lat_max) & 
                                         (self.earthquake_db["longitude"] >= lon_min) & (self.earthquake_db["longitude"] <= lon_max)]
        if filtered_df.empty:
            earthquake_history = "No earthquakes with a magnitude greater than 4 have occurred within the given range since 1970."
        else:
            selected_columns = ["time", "latitude", "longitude", "place", "mag", "magType", "depth", "type", "updated", "gap"]
            earthquake_history = filtered_df[selected_columns].to_dict(orient="records")

        return earthquake_history

if __name__ == "__main__":
    min_lon = 113.00  # E
    max_lon = 114.00  # E
    min_lat = 34.33  # N
    max_lat = 35.00  # N
    history_earthquake_db = history_earthquake_db()
    print(history_earthquake_db.get_earthquake_history(min_lon, min_lat, max_lon, max_lat))
