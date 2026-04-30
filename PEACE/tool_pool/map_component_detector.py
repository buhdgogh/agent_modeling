import os
os.sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/..")
from PEACE.dependencies.ultralytics import YOLOv10

class map_component_detector:
    def __init__(self, model_path=os.path.join(os.path.dirname(__file__), "..", "models", "det_component", "weights", "best.pt")):
        self.model = YOLOv10(model_path)

    def detect(self, image_path):
        objs = self.model.predict(source=image_path)[0]
        return objs

if __name__ == "__main__":
    map_component_detector = map_component_detector()
    print(map_component_detector.detect(r"E:\Code\geo\geo\PEACE\images\sample_cgs.jpg"))
