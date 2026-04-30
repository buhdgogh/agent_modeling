import os
os.sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}")
from .map_component_detector import map_component_detector
from .map_legend_detector import map_legend_detector
from .k2_knowledge_db import k2_knowledge_db, geological_knwoledge_type
# from .active_fault_db import active_fault_db
# from .history_earthquake_db import history_earthquake_db
# from .landcover_type_api import landcover_type_api
# from .population_density_api import population_density_api
# from rock_type_and_age_db import rock_type_and_age_db

__all__ = ["map_component_detector", "map_legend_detector", "k2_knowledge_db", "geological_knwoledge_type"]
