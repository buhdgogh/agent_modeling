import os
import json
from utils import api, prompt, vision, common
from modules import hierarchical_information_extraction, domain_knowledge_injection, prompt_enhanced_QA

hie = hierarchical_information_extraction()
dki = domain_knowledge_injection()
peqa = prompt_enhanced_QA()

def copilot(image_path, question, question_type, copilot_modes=["HIE", "DKI", "PEQA"]):
    # Content filter.
    if common.rai_filter(question): # debug
        return "I can't help you with that."

    # Hierarchical information extraction module.
    if "HIE" in copilot_modes:
        information = hie.digitalize(image_path)
    else:
        information = None

    # Domain knowledge injection module.
    if "DKI" in copilot_modes:
        knowledge = dki.consult(question, information)
    else:
        knowledge = None

    # Prompt-enhanced QA module.
    if "PEQA" in copilot_modes:
        answer = peqa.answer(information, knowledge, True, image_path, question, question_type)
    else:
        answer = peqa.answer(information, knowledge, False, image_path, question, question_type)

    try:
        answer = json.loads(answer)
        final_answer = prompt.get_final_answer(answer, question_type)
    except:
        final_answer = answer

    if common.echo:
        print("Selected knowledge:", list(knowledge.keys()) if knowledge else knowledge)
        print("Raw Answer:", answer)
        print("Final Answer:", final_answer)
        print("======================================================")
    return final_answer


if __name__ == "__main__":
    image_path = "images/sample_usgs.jpg"
    question = "What is the title of this map?"
    question_type = "extracting-sheet_name"
    copilot_modes = ["HIE", "DKI", "PEQA"]
    answer = copilot(image_path, question, question_type, copilot_modes)
    print(f"Question: {question}\nAnswer: {answer}\n")
