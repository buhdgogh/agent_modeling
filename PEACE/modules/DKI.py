import os
os.sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/..")
import json
from ..utils import api, prompt, vision, common
from ..agents import geographer_agent, seismologist_agent

class domain_knowledge_injection:
    def __init__(self):
        self.seismologist = seismologist_agent()
        self.geographer = geographer_agent()

    def select(self, question, knowledge):
        knowledge_types = list(knowledge.keys())
        examples = '{"required_knowledge_types": %s}' % knowledge_types
        instructions = [
            {"type": "text", "text": f"The given question is '{question}'."},
            {"type": "text", "text": f"The knowledge types from expert group are {', '.join(knowledge_types)}."},
            {"type": "text", "text": f'What are the helpful knowledge types among them to answer the given question, the example is {examples}, only respond with JSON format.\n'},
        ]
        messages = [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": instructions},
        ]
        answer = api.answer_wrapper(messages, structured=True)
        try:
            answer = eval(answer)
            keys = answer["required_knowledge_types"]
        except:
            keys = list()

        selected_knowledge = dict()
        for key in keys:
            if key in knowledge:
                selected_knowledge[key] = knowledge[key]
        return selected_knowledge

    def consult(self, question, meta):
        if meta is None:
            print("Missing metadata in DKI module", flush=True)
            return None

        knowledge_path = os.path.join(common.cache_path(), "knowledge", meta["name"] + ".json")
        if os.path.exists(knowledge_path):
            knowledge = json.loads(open(knowledge_path).read())
        else:
            longitude_range = meta["information"]["longitude"]
            longitude_range = list(map(lambda x: common.convert_to_decimal(x), longitude_range))
            latitude_range = meta["information"]["latitude"]
            latitude_range = list(map(lambda x: common.convert_to_decimal(x), latitude_range))
            min_lon = min(longitude_range)
            max_lon = max(longitude_range)
            min_lat = min(latitude_range)
            max_lat = max(latitude_range)
            if common.is_valid_longitude(min_lon) and \
            common.is_valid_longitude(max_lon) and \
            common.is_valid_latitude(min_lat) and \
            common.is_valid_latitude(max_lat):
                seismic_data = self.seismologist.get_knowledge(min_lon, min_lat, max_lon, max_lat)
                geographical_data = self.geographer.get_knowledge(min_lon, min_lat, max_lon, max_lat)
                knowledge = seismic_data | geographical_data
            else:
                knowledge = dict()
            
            # output external knowledge of geologic map.
            common.create_folder_by_file_path(knowledge_path)
            with open(knowledge_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(knowledge, indent=4, ensure_ascii=False))
        
        selected_knowledge = self.select(question, knowledge)
        return selected_knowledge


if __name__ == "__main__":
    meta = {
        "name": "E4901",
        "information": {
            "longitude": [
                "120°00'E",
                "121°00'E"
            ],
            "latitude": [
                "30°30'N",
                "31°30'N"
            ]
        }
    }
    question = "Based on this geologic map, please analyze the seismic risk level in this area?"
    dki = domain_knowledge_injection()
    external_knowledge = dki.consult(question, meta)
    print(external_knowledge)
