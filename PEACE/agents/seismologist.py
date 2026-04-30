import os
os.sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/..")
import json
from ..utils import api, prompt, vision, common
from ..tool_pool import history_earthquake_db
from ..tool_pool import active_fault_db

class seismologist_agent:
    def __init__(self):
        self.active_fault_db = active_fault_db()
        self.history_earthquake_db = history_earthquake_db()

    def get_knowledge(self, min_lon, min_lat, max_lon, max_lat):
        active_faults = self.active_fault_db.get_active_faults(min_lon, min_lat, max_lon, max_lat)
        earthquake_history = self.history_earthquake_db.get_earthquake_history(min_lon, min_lat, max_lon, max_lat)
        seismic_data = {
            "active_faults": active_faults,
            "earthquake_history": earthquake_history,
        }
        return seismic_data

if __name__ == "__main__":
    min_lon = 108.00#E
    max_lon = 109.00#E
    min_lat = 19.33#N
    max_lat = 20.00#N
    seismologist_agent = seismologist_agent()
    print(seismologist_agent.get_knowledge(min_lon, min_lat, max_lon, max_lat))
