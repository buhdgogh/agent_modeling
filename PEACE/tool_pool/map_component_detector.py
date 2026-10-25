import os
from PEACE.dependencies.ultralytics import YOLOv10

class map_component_detector:
    def __init__(self, model_path=os.path.join(os.path.dirname(__file__), "..", "models", "det_component", "weights", "best.pt")):
        self.model = YOLOv10(model_path)

    def detect(self, image_path):
        objs = self.model.predict(source=image_path)[0]
        return objs
