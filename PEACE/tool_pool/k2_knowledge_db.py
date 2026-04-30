import os
import json
from enum import Enum
from deep_translator import GoogleTranslator
from transformers import AutoModel
from sentence_transformers import util

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class geological_knwoledge_type(Enum):
    Component_Usage = 1
    Downstream_Task = 2
    Rock_Type = 3
    Rock_Age = 4
    Rock_Des = 5

class k2_knowledge_db:
    def __init__(self):
        self.embedding_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)

        # load k2 expert knowledge base of rock type
        with open(os.path.join(os.path.dirname(__file__), "..", "dependencies", "knowledge", "k2_rock_type.json"), "r") as f:
            rock_type_kb = json.load(f)
        self.rock_type_kb = {item["rock_value"]: item["rock_name"] for item in rock_type_kb}
        self.rock_type_embeddings = self.embedding_model.encode(list(self.rock_type_kb.keys()), task="text-matching")

        # load k2 expert knowledge base of rock age
        with open(os.path.join(os.path.dirname(__file__), "..", "dependencies", "knowledge", "k2_rock_age.json"), "r") as f:
            rock_age_kb = json.load(f)
        self.rock_age_kb = {item["rock_value"]: item["rock_name"] for item in rock_age_kb}
        self.rock_age_embeddings = self.embedding_model.encode(list(self.rock_age_kb.keys()), task="text-matching")

        # load k2 expert knowledge base of rock detail
        with open(os.path.join(os.path.dirname(__file__), "..", "dependencies", "knowledge", "k2_rock_detail.json"), "r") as f:
            rock_detail_kb = json.load(f)
        self.rock_detail_kb = {item["id"]: item["answer"] for item in rock_detail_kb}
        self.rock_detail_embeddings = self.embedding_model.encode(list(self.rock_detail_kb.keys()), task="text-matching")

        # load k2 expert knowledge base of component usage
        with open(os.path.join(os.path.dirname(__file__), "..", "dependencies", "knowledge", "k2_usage.json"), "r") as f:
            component_usage_kb = json.load(f)
        self.component_usage_kb = {item["question"]: item["answer"] for item in component_usage_kb}
        self.component_usage_key_embeddings = self.embedding_model.encode(list(self.component_usage_kb.keys()), task="text-matching")

        # load k2 expert knowledge base of downstream task
        with open(os.path.join(os.path.dirname(__file__), "..", "dependencies", "knowledge", "k2_expertise.json"), "r") as f:
            downstream_task_kb = json.load(f)
        self.downstream_task_kb = {item["question"]: item["answer"] for item in downstream_task_kb}
        self.downstream_task_key_embeddings = self.embedding_model.encode(list(self.downstream_task_kb.keys()), task="text-matching")

    def semantic_match(self, key_embeddings, d, target, threshold):
        keys = list(d.keys())
        target_embedding = self.embedding_model.encode(target, task="text-matching")
        similarities = util.cos_sim(target_embedding, key_embeddings)[0]
        matched_keys = [keys[i] for i in range(len(keys)) if similarities[i] >= threshold]
        return {key: d[key] for key in matched_keys}

    def get_rock_type(self, query):
        query = GoogleTranslator(source="auto", target="en").translate(query)
        rock_type = self.semantic_match(self.rock_type_embeddings, self.rock_type_kb, query, threshold=0.85)
        return rock_type

    def get_rock_age(self, query):
        query = GoogleTranslator(source="auto", target="en").translate(query)
        rock_age = self.semantic_match(self.rock_age_embeddings, self.rock_age_kb, query, threshold=0.85)
        return rock_age

    def get_rock_knowledge(self, query):
        query = GoogleTranslator(source="auto", target="en").translate(query)
        rock_detail = self.semantic_match(self.rock_detail_embeddings, self.rock_detail_kb, query, threshold=0.85)
        return rock_detail

    def get_component_usage_knowledge(self, query):
        query = "What is the function of " + GoogleTranslator(source="auto", target="en").translate(query) + " in geologic maps?"
        component_usage_knowledge = self.semantic_match(self.component_usage_key_embeddings, self.component_usage_kb, query, threshold=0.85)
        return component_usage_knowledge

    def get_downstream_task_knowledge(self, query):
        query = "How do geologists conduct the task of " + GoogleTranslator(source="auto", target="en").translate(query) + "?"
        downstream_task_knowledge = self.semantic_match(self.downstream_task_key_embeddings, self.downstream_task_kb, query, threshold=0.85)
        return downstream_task_knowledge

if __name__ == "__main__":
    k2_knowledge_db = k2_knowledge_db()
    print(k2_knowledge_db.get_knowledge(geological_knwoledge_type.Component_Usage, "图例"))
    print(k2_knowledge_db.get_knowledge(geological_knwoledge_type.Downstream_Task, "获取当前区域的邻接信息"))
    print(k2_knowledge_db.get_knowledge(geological_knwoledge_type.Rock, "沉积岩"))
