import os
os.sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/..")
import cv2
import json
from ..utils import api, prompt, vision, common
# from ..tool_pool import k2_knowledge_db, geological_knwoledge_type
from ..tool_pool import map_component_detector
from ..tool_pool import map_legend_detector
# from ..tool_pool import rock_type_and_age_db

class geologist_agent:
    def __init__(self):
        self.map_component_detector = map_component_detector()
        self.map_legend_detector = map_legend_detector()
        # self.k2_knowledge_db = k2_knowledge_db()
        self.rock_type_db = rock_type_and_age_db("type")
        self.rock_age_db = rock_type_and_age_db("age")

    def get_map_layout(self, map_image_path):
        components = self.map_component_detector.detect(map_image_path)
        map_layout = {"regions": components}
        return map_layout

    def get_legend_metadata(self, legend_image_path, legend_bndbox):
        legend_image = cv2.imread(legend_image_path)
        legends = self.map_legend_detector.detect(legend_image_path)

        # calculate text bndbox in legend.
        legend_units = legends.values()

        # calculate color of each color bndbox.
        self.extract_legend_color(legend_image, legend_units)

        # ocr of each text bndbox.
        self.extract_legend_text(legend_image, legend_units)

        # rectify color bndbox in legend.
        for legend in legend_units:
            for name in ["color_bndbox", "text_bndbox"]:
                x0, y0, x1, y1 = legend[name]
                x0 += legend_bndbox[0]
                y0 += legend_bndbox[1]
                x1 += legend_bndbox[0]
                y1 += legend_bndbox[1]
                legend[name] = [x0, y0, x1, y1]

        legend_metadata = {"legend": legends}
        return legend_metadata

    def extract_legend_color(self, image, legends):
        for legend in legends:
            bndbox = legend.get("color_bndbox", list())
            if len(bndbox) == 0 or not common.is_valid_bndbox(*bndbox, image.shape[1], image.shape[0]):
                legend["color"] = [255, 255, 255]
                legend["color_name"] = "White"
                continue
            cropped_image = vision.crop_image(image, bndbox)
            color = vision.calc_image_rgb(cropped_image)
            legend["color"] = list(map(int, color))
            legend["color_name"] = vision.rgb_to_color_name(color)

    def extract_legend_text(self, image, legends):
        h, w, _ = image.shape
        for legend in legends:
            x0, y0, x1, y1 = legend["text_bndbox"]
            if not common.is_valid_bndbox(x0, y0, x1, y1, w, h):
                legend["text"] = "unknown"
                continue
            unit_image = image[y0:y1, x0:x1]
            unit_image_rgb = cv2.cvtColor(unit_image, cv2.COLOR_BGR2RGB)
            #text = vision.image_ocr(unit_image)
            instructions = list()
            instructions.append({"type": "image_url", "image_url": {"url": api.input_image_to_data_url(unit_image_rgb)}})
            instructions.append({"type": "text", "text": "Only output the OCR result of the given image."})
            messages = [
                {"role": "system", "content": "You are an OCR expert."},
                {"role": "user", "content": instructions},
            ]
            text = api.answer_wrapper(messages, structured=False)
            if text is not None:
                text = text.split(":")[-1].split("：")[-1].strip().strip("-")
            legend["text"] = text
    
    def get_knowledge(self, type, query):
        if type == geological_knwoledge_type.Rock_Type:
            rock_type = self.rock_type_db.get_rock_type_or_age(query)
            geological_knowledge = {"rock_type": rock_type}
        elif type == geological_knwoledge_type.Rock_Age:
            rock_age = self.rock_age_db.get_rock_type_or_age(query)
            geological_knowledge = {"rock_age": rock_age}
        elif type == geological_knwoledge_type.Component_Usage:
            component_usage_knowledge = self.k2_knowledge_db.get_component_usage_knowledge(query)
            geological_knowledge = {"component_usage_knowledge": component_usage_knowledge}
        elif type == geological_knwoledge_type.Downstream_Task:
            downstream_task_knowledge = self.k2_knowledge_db.get_downstream_task_knowledge(query)
            geological_knowledge = {"downstream_task_knowledge": downstream_task_knowledge}
        elif type == geological_knwoledge_type.Rock_Des:
            rock_knowledge = self.k2_knowledge_db.get_rock_knowledge(query)
            geological_knowledge = {"rock_knowledge": rock_knowledge}
        return geological_knowledge


if __name__ == "__main__":
    image_path = "sample.jpg"
    geologist_agent = geologist_agent()
    map_layout = geologist_agent.get_map_layout(image_path)
    print(map_layout)

    legend_path = "legend.jpg"
    image = cv2.imread(image_path)
    legend_bndbox = map_layout["regions"]["legend"][0]
    common.create_folder_by_file_path(legend_path)
    vision.crop_and_save_image(image, legend_bndbox, legend_path)
    legend_metadata = geologist_agent.get_legend_metadata(legend_path, legend_bndbox)
    print(legend_metadata)

    #print(geologist_agent.get_knowledge(geological_knwoledge_type.Component_Usage, "图例"))
    #print(geologist_agent.get_knowledge(geological_knwoledge_type.Downstream_Task, "获取当前区域的邻接信息"))
    #print(geologist_agent.get_knowledge(geological_knwoledge_type.Rock, "沉积岩"))
