import os
os.sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/..")
import json
from ..utils import api, prompt, vision, common

class prompt_enhanced_QA:
    def __init__(self):
        relation_path = os.path.join(common.cache_path(), "component", "relations.json")
        self.components = list(prompt.components)
        if os.path.exists(relation_path):
            self.component_relations = json.loads(open(relation_path).read())
        else:
            examples = [
                {"component1": "main_map", "component2": "legend", "relation": "XXX"},
                {"component1": "scale", "component2": "title", "relation": "XXX"},
            ]
            instructions = [
                {"type": "text", "text": f"The components of geologic map are {', '.join(self.components)}."},
                {"type": "text", "text": f'What are the relations for all the component pairs in geologic map, the example is {examples}, only respond with JSON format.\n'},
            ]
            messages = [
                {"role": "system", "content": prompt.system_prompt},
                {"role": "user", "content": instructions},
            ]
            self.component_relations = api.answer_wrapper(messages, structured=True)

            # output component relations of geologic map.
            common.create_folder_by_file_path(relation_path)
            with open(relation_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(self.component_relations, indent=4, ensure_ascii=False))

    def select(self, question, question_type):
        component_path = os.path.join(common.cache_path(), "component", question_type + ".json")
        if os.path.exists(component_path):
            selected_components = json.loads(open(component_path).read())
            return selected_components

        examples = {"1": "XXX", "2": "XXX"}
        instructions = [
            {"type": "text", "text": f"The component relations of geologic map are {self.component_relations}"},
            {"type": "text", "text": f"Select component(s) from {', '.join(self.components)}, which is/are used to answer the question, {question}, the example is: {examples}, only respond with JSON format by the order of importance.\n"},
        ]
        messages = [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": instructions},
        ]
        answer = api.answer_wrapper(messages, structured=True)
        try:
            selected_components = eval(answer)
            selected_components = list(map(lambda x: x[1], sorted(selected_components.items(), key=lambda x: x[0])))
        except:
            selected_components = None

        # output selected components of geologic map.
        common.create_folder_by_file_path(component_path)
        with open(component_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(selected_components, indent=4, ensure_ascii=False))
        return selected_components
    
    def answer(self, information, knowledge, enhance_prompt, image_path, question, question_type):
        instructions = list()

        # Context enhancement.
        if information is not None:
            prompt.polish_information(information)
            instructions.append({"type": "text", "text": str(information)})
        if knowledge is not None:
            instructions.append({"type": "text", "text": str(knowledge)})

        if enhance_prompt:
            # Component selection.
            selected_components = self.select(question, question_type)
            if selected_components is not None:
                if information is not None:
                    for selected_component in selected_components:
                        selected_image_path = os.path.join(common.cache_path(), "det", information["name"], f"{selected_component}_0.png")
                        if os.path.exists(selected_image_path):
                            instructions.append({"type": "image_url", "image_url": {"url": api.local_image_to_data_url(selected_image_path)}})
                    if len(instructions) == 0:
                        if len(selected_components) > 0:
                            instructions.append({"type": "text", "text": f"Let's focus more on {', '.join(selected_components)}"})
                        instructions.append({"type": "image_url", "image_url": {"url": api.local_image_to_data_url(image_path)}})
                else:
                    if len(selected_components) > 0:
                        instructions.append({"type": "text", "text": f"Let's focus more on {', '.join(selected_components)}"})
                    instructions.append({"type": "image_url", "image_url": {"url": api.local_image_to_data_url(image_path)}})
            # Both COT and JSON format + few-shot.
            question_instruction = prompt.ability2instruction(question_type, vision.image_size(image_path))
            instructions.append({"type": "text", "text": f"Instruction: {question_instruction}\n"})
        else:
            instructions.append({"type": "image_url", "image_url": {"url": api.local_image_to_data_url(image_path)}})
            # JSON format + few-shot.
            question_instruction = prompt.ability2instruction(question_type, vision.image_size(image_path))
            question_instruction = question_instruction[:question_instruction.find('{"answer":')] + '{"answer": "XXX"}'
            instructions.append({"type": "text", "text": f"Instruction: {question_instruction}\n"})

        instructions.append({"type": "text", "text": f"Question: {question}\n"})
        instructions.append({"type": "text", "text": f"Answer: "})

        messages = [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": instructions},
        ]
        if common.echo:
            print("======================================================")
            print("Image:", image_path, flush=True)
            print("Question Type:", question_type, flush=True)
            print("Question Instruction:", question_instruction, flush=True)
            print("Question:", question, flush=True)
        answer = api.answer_wrapper(messages, structured=True)
        return answer


if __name__ == "__main__":
    question = "请你推断 (经度:109.0, 纬度:19.67) 所在的地质图的图幅名称？"
    question_type = "extracting-sheet_name"
    peqa = prompt_enhanced_QA()
    selected_components = peqa.select(question, question_type)
    print(selected_components)
