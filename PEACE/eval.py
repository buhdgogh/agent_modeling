import os
import argparse
import pandas as pd
from tqdm import tqdm
from copilot import copilot
from utils import prompt, common

def eval_copilot(args, image_folder, q_path, qa_path, overwrite=False):
    # load data.
    bench_qa = pd.read_csv(qa_path if not overwrite and os.path.exists(qa_path) else q_path, index_col=False)
    bench_qa = bench_qa.sort_values(by="img_path").reset_index(drop=True)
    # set placeholder for model answer.
    if "model_answer" not in bench_qa.columns:
        bench_qa["model_answer"] = ""

    # answer question.
    for i, meta in tqdm(bench_qa.iterrows(), total=len(bench_qa)):
        image_path = os.path.join(image_folder, os.path.basename(meta["img_path"]))
        question = meta["question"]
        question = prompt.remove_format_requirement(question)
        question = prompt.format_question(question, meta)
        question_type = meta["type"]
        
        try:
            model_answer = copilot(image_path, question, question_type, args.copilot_mode.split(","))
        except:
            model_answer = ""
        bench_qa.loc[i, "model_answer"] = model_answer

    # save model answer.
    new_order = ["img_path", "type", "question", "answer", "model_answer", "mcq", "A", "B", "C", "D"]
    bench_qa = bench_qa[new_order]
    bench_qa.to_csv(qa_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GeoMap-Agent: evaluation script")
    #parser.add_argument("--copilot_mode", type=str, default="None", help="The copilot mode: None, HIE, DKI, PEQA")
    parser.add_argument("--copilot_mode", type=str, default="HIE,DKI,PEQA", help="The copilot mode: None, HIE, DKI, PEQA")
    parser.add_argument("--dataset_source", type=str, default="usgs", help="The dataset source: usgs, cgs")
    args = parser.parse_args()

    common.dataset_source = args.dataset_source
    image_folder = f"./data/{common.dataset_source}_images"
    q_path = f"./data/{common.dataset_source}_qas.csv"
    qa_path = f"./{common.dataset_source}_{common.model_name}_{args.copilot_mode.replace(',', '-')}_answer.csv"
    eval_copilot(args, image_folder, q_path, qa_path, overwrite=True)
