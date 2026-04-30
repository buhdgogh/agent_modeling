import os
os.sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}")
from .geologist import geologist_agent
from .geographer import geographer_agent
from .seismologist import seismologist_agent

__all__ = ["geologist_agent", "geographer_agent", "seismologist_agent"]
