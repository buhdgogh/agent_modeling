import os
import re
import time
import base64
import logging
import argparse
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from mimetypes import guess_type
from collections import defaultdict
import openai
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

azure_endpoint = ""  # to be filled.
identity_id = ""  # to be filled.
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(managed_identity_client_id=identity_id),
    "https://cognitiveservices.azure.com/.default")
api_version = "2024-08-01-preview"
client = AzureOpenAI(
    azure_endpoint=azure_endpoint,
    azure_ad_token_provider=token_provider,
    api_version=api_version,
    max_retries=0,
)

def answer_wrapper(messages, model_name="4o", max_tks=2048, temperature=0.0, structured=False):
    answer = None
    response = None
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tks,
            response_format={"type": "json_object" if structured else "text"},
        )
        answer = response.choices[0].message.content
    # https://github.com/openai/openai-python/blob/main/openai/error.py
    except openai.RateLimitError as e:
        logging.warning(f"RateLimitError: {e}. Retrying...")
        time.sleep(2)
    except openai.BadRequestError as e:
        if e.code == "content_filter":
            is_valid_response = False
            logging.warning(f"BadRequestError:content_filter: {e}. Skipping...")
        elif e.code == "context_length_exceeded":
            is_valid_response = False
            logging.warning(f"BadRequestError:context_length_exceeded: {e}. Skipping...")
        else:
            logging.warning(f"BadRequestError: {e}. Skipping...")
            is_valid_response = False
    except openai.APITimeoutError as e:
        logging.warning(f"APITimeoutError: {e}. Retrying...")
        time.sleep(2)
    except openai.APIConnectionError as e:
        logging.warning(f"Connection aborted: {e}.")
        time.sleep(2)
    except Exception as e:
        if response is not None and response.choices[0].finish_reason == "content_filter":
            print(messages)
            print(response.choices[0].content_filter_results)
            logging.warning(f"{type(e).__name__}:content_filter: {e}. Skipping...")
            is_valid_response = False
        else:
            logging.warning(f"{type(e).__name__}: {e}. Retrying...")
            time.sleep(2)

    return answer

def get_iou(bbox1, bbox2):
    x0 = max(bbox1[0], bbox2[0])
    y0 = max(bbox1[1], bbox2[1])
    x1 = min(bbox1[2], bbox2[2])
    y1 = min(bbox1[3], bbox2[3])
    if x1 < x0 or y1 < y0:
        return 0.0
    
    inter_area = (x1 - x0) * (y1 - y0)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - inter_area
    iou = inter_area / union_area
    return iou

def convert_to_decimal(degree_str):
    match = re.match(r"(\d+)°(\d+)'[(\d+)(\"|'+)]*([NSWE])", degree_str.strip())
    if not match:
        raise ValueError(f"Invalid format: {degree_str}")

    degrees = int(match.group(1))
    minutes = int(match.group(2))
    direction = match.group(3)
    decimal_degrees = degrees + minutes / 60.0
    if direction in ["S", "W"]:
        decimal_degrees = -decimal_degrees

    return decimal_degrees

def get_lonlat_range(answer):
    lon_range, lat_range = answer.split(",")

    lon_range = lon_range.split("-")
    lon_range = [convert_to_decimal(lon) for lon in lon_range]
    lon_range.sort()

    lat_range = lat_range.split("-")
    lat_range = [convert_to_decimal(lat) for lat in lat_range]
    lat_range.sort()
    return lon_range, lat_range

def local_image_to_data_url(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_encoded_data}"

def judge_essay(image_path, answer, model_answer):
    score = 0
    ans_pairs = (
        (answer, model_answer),
        (model_answer, answer),
    )
    for ans1, ans2 in ans_pairs:
        original_question = "based on this geologic map, please analyze the seismic risk level in this area?"
        instruction = f"\
To evaluate which of the two answers is better for the question that {original_question}, consider the following criteria:\
1. Diversity: The answer should address various aspects of the question, providing a well-rounded perspective.\
2. Specificity: The answer should be detailed and precise, avoiding vague or general statements.\
3. Professionalism: The answer should be articulated in a professional manner, demonstrating expertise and credibility.\
"
        question = f"\
Answer1:\
{ans1}\
Answer2:\
{ans2}\
Question: which answer is better?\
A. Answer1 is better\
B. Answer2 is better\
C. Tie\
"
        format = '\
Only respond A, B or C in JSON format, for example: {"answer": "C"}\
Answer: \
'
        prompt = [
            {"type": "image_url", "image_url": {"url": local_image_to_data_url(image_path)}},
            {"type": "text", "text": instruction},
            {"type": "text", "text": question},
            {"type": "text", "text": format},
        ]
        messages = [
            {"role": "system", "content": "You are an experienced geologist, and you are skilled at judging the quality of earthquake risk assessments."},
            {"role":   "user", "content": prompt},
        ]

        try:
            judge_answer = answer_wrapper(messages, model_name="4o", structured=True)
            #print("===================")
            #print(judge_answer)
            judge_answer = eval(judge_answer)
            judge_answer = judge_answer["answer"]
        except:
            judge_answer = "C"
        
        if judge_answer == "C":
            score += 0.5 / len(ans_pairs)
        elif (judge_answer == "A" and ans1 == answer) or \
                (judge_answer == "B" and ans2 == answer):
            score += 1.0 / len(ans_pairs)
    return score

def filter_answer(model_answer):
    if pd.isna(model_answer) or model_answer is None:
        return True
    if model_answer == "TBD" or model_answer == "TBF" or model_answer == "":
        assert False
        return True
    if model_answer == "No Answer" or model_answer == "No Answer, Error in Agent" or model_answer == "No Answer, Error in API":
        assert False
        return True
    return False

def calculate_metrics(qa_path, dataset_source):
    data = pd.read_csv(qa_path)
    print("-------------------------------")
    print(f"overall: {len(data)}")
    
    qt_pass_cnt = dict()
    for key in list(data["type"].unique()):
        qt_pass_cnt[key] = 0
    qt_total_cnt = dict(data["type"].value_counts())
    for key, value in qt_total_cnt.items():
        print(f"{key}: {value}")
    print("-------------------------------")

    white_qt_list = None
    #white_qt_list = ("reasoning-area_comparison",)
    black_qt_list = []
    #black_qt_list = ("analyzing-earthquake_risk",)

    for idx, row in tqdm(data.iterrows(), total=len(data)):
        image_path = os.path.join(f"./data/{dataset_source}_images", row["img_path"])
        qt = row["type"]
        question = row["question"]
        answer = row["answer"]
        model_answer = row["model_answer"]

        if white_qt_list is not None and qt not in white_qt_list:
            continue
        if qt in black_qt_list:
            continue
        
        # filter
        if filter_answer(model_answer):
            #print(model_answer)
            qt_total_cnt[qt] -= 1
            continue

        try:
            # multiple choice question.
            if row["mcq"]:
                answer_choice = re.findall(r"[ABCD]", model_answer)[0]
                if answer_choice == answer:
                    qt_pass_cnt[qt] += 1
            # yes/no question.
            elif "reasoning-fault_existence" in qt:
                pattern = f"^{answer}.*"
                if re.search(pattern, model_answer, re.IGNORECASE):
                    qt_pass_cnt[qt] += 1
            # essay question.
            elif "analyzing-earthquake_risk" in qt:
                score = judge_essay(image_path, answer, model_answer)
                qt_pass_cnt[qt] += score
            # fill-in-the-blank question.
            else:
                # general questions.
                if answer.strip().lower() == model_answer.strip().lower():
                    qt_pass_cnt[qt] += 1
                else:
                    # all grounding questions.
                    if "grounding-" in qt:
                        # debug
                        #s = model_answer.find("[")
                        #t = model_answer.find("]")
                        #model_answer = model_answer[s:t+1].replace("[[", "[").replace("]]", "]")
                        ############

                        model_answer = list(map(float, eval(model_answer)))
                        answer = eval(answer)
                        if get_iou(model_answer, answer) >= 0.5:
                            qt_pass_cnt[qt] += 1
                    # index map extracting question.
                    if "extracting-index_map" in qt:
                        model_answer = set(eval(model_answer))
                        answer = set(eval(answer))
                        intersection_ratio = len(model_answer & answer) / len(model_answer | answer) if model_answer | answer else 0
                        if intersection_ratio >= 0.5:
                            qt_pass_cnt[qt] += 1
                    # all grounding questions.
                    if "extracting-lonlat" in qt:
                        lon_gt, lat_gt = get_lonlat_range(answer)
                        lon_pv, lat_pv = get_lonlat_range(model_answer)
                        thred = 1.0 / 60.0
                        if abs(lon_gt[0] - lon_pv[0]) <= thred:
                            qt_pass_cnt[qt] += 0.25
                        if abs(lon_gt[1] - lon_pv[1]) <= thred:
                            qt_pass_cnt[qt] += 0.25
                        if abs(lat_gt[0] - lat_pv[0]) <= thred:
                            qt_pass_cnt[qt] += 0.25
                        if abs(lat_gt[1] - lat_pv[1]) <= thred:
                            qt_pass_cnt[qt] += 0.25
        except:
            #print(f"question type: {qt}\nquestion: {question}\nanswer: {answer}\nmodel_answer: {model_answer}\n")
            #traceback.print_exc()
            pass

    result = dict()
    pass_cnt = 0
    total_cnt = 0
    for qt in qt_pass_cnt:
        if qt in qt_total_cnt:
            result[qt] = qt_pass_cnt[qt] / qt_total_cnt[qt]
            pass_cnt += qt_pass_cnt[qt]
            total_cnt += qt_total_cnt[qt]
    result["overall"] = pass_cnt / total_cnt
    return result

def dump_result(args, result):
    # details.
    pd.Series(result).plot(kind="bar", color="blue")
    pd.Series(result).to_json(f"./summary/{args.dataset_source}/details_{args.model_name}.json", orient="index", indent=4)
    print("---------------------------------")
    print(pd.Series(result).sort_index())

    # abilities.
    grouped = defaultdict(list)
    for key, value in result.items():
        prefix = key.split("-")[0]
        grouped[prefix].append(value)
    averages = pd.Series(grouped).apply(lambda x: np.mean(x))
    averages.plot(kind="bar", color="blue")
    averages.to_json(f"./summary/{args.dataset_source}/abilities_{args.model_name}.json", orient="index", indent=4)
    print("---------------------------------")
    print(averages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script of calculating measurement metrics.")
    parser.add_argument("--model_name", type=str, default="4o", help="The model name: 4o, 4o-mini, qwen_chat, idefics_9b_instruct, glm-4v-9b, cogvlm2-llama3-chat-19B, monkey-chat")
    parser.add_argument("--copilot_mode", type=str, default="HIE,DKI,PEQA", help="The copilot mode: HIE, PEQA, DKI")
    parser.add_argument("--dataset_source", type=str, default="usgs", help="The dataset source: cgs, usgs")

    args = parser.parse_args()
    qa_path = f"./{args.dataset_source}_{args.model_name}_{args.copilot_mode.replace(',', '-')}_answer.csv"
    #print(args)
    result = calculate_metrics(qa_path, args.dataset_source)
    dump_result(args, result)
