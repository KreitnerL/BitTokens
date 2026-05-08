import argparse
import datetime
import os
from pathlib import Path
from sys import maxsize

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig
from tqdm.auto import tqdm

from utils import (
    AI_MESSAGE,
    MAX_NEW_TOKENS,
    SYSTEM_MESSAGE,
    eval_regression,
    parse_response,
)

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--number_file_path", type=str, help="Path to CSV file with numbers")
parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
parser.add_argument("--model_name", type=str, default="gemini-2.5-pro-preview-05-06", help="Model name")
parser.add_argument("--offset", type=int, default=0, help="Start index of the dataset")
parser.add_argument("--chunk_id", type=int, default=0, help="Chunk ID")
parser.add_argument("--chunk_size", type=int, default=maxsize, help="Chunk size")
parser.add_argument("--save_interval", type=int, default=100, help="Number of samples after which to sync to disk")
parser.add_argument("--regression", action="store_true", help="Use regression evaluation")
parser.add_argument("--acceptance_threshold", type=float, default=0.99, help="Acceptance threshold for regression")
parser.add_argument("--verbose", action="store_true", help="Print verbose output")
args = parser.parse_args()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load dataset
NAME = Path(args.number_file_path).stem
df = pd.read_csv(args.number_file_path, dtype=str, skiprows=range(1,args.chunk_id * args.chunk_size + args.offset + 1), nrows=args.chunk_size)
EXAMPLE = f"{df.loc[0, 'text_prompt']}\n{AI_MESSAGE} {df.loc[0, 'answer']}"
df.drop_duplicates(subset=["text_prompt"], inplace=True)

results = list()
results_dir = Path(args.results_dir) / args.model_name
results_dir.mkdir(parents=True, exist_ok=True)

current_date_and_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
file_path = f"{results_dir}/{NAME}_{current_date_and_time}_{args.chunk_id}.csv"

tq = tqdm(df.to_dict(orient="records"), desc="sample", leave=False) if args.verbose else df.to_dict(orient="records")
for row_idx, row_dict in enumerate(tq):
    if args.verbose:
        tq.set_description(row_dict['text_prompt'])

    gen_config = GenerateContentConfig(
        system_instruction=SYSTEM_MESSAGE+EXAMPLE,
        max_output_tokens=MAX_NEW_TOKENS*100, # gemini-2.5-pro-preview-03-25 has issues with max_output_tokens
        thinking_config=ThinkingConfig(thinking_budget=0) if args.model_name == "gemini-2.5-pro-preview-03-25" else None,  # Turn of reasoning
        temperature=0.0,
    )
    prompt = f"{row_dict['text_prompt']}\n{AI_MESSAGE}"
    # Add retry logic for API overload
    retry_count = 0
    max_retries = 10
    response: str = None
    while retry_count < max_retries:
        try:
            response = client.models.generate_content(
                model=args.model_name,
                contents=prompt,
                config=gen_config
            )
            response = response.text.replace("\n", "").strip()
            retry_count = 0 # Reset retry count on success
            break
        
        except Exception as e:
            if "503 UNAVAILABLE" in str(e) or "502 Bad Gateway" in str(e):
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Model overloaded. Retry {retry_count}/{max_retries} after waiting 60s...")
                    import time
                    time.sleep(60)  # Wait for 60 seconds before retrying
                    continue
                else:
                    print(f"Failed after {max_retries} retries due to model overload.")
            if len(results)>0:
                print(f"Flushing results to disk {file_path}")
                results_df = pd.DataFrame(results)
                results_df.to_csv(file_path, index=False, mode="a", header=False)
                acc = pd.read_csv(file_path).correct.mean()
                print(f"Accuracy: {acc:.2%}")
            print(f"Error with response: {response if (isinstance(response, str) or response is None) else response.model_dump_json()}")
            raise e

    if args.regression:
        correct, sMAPE_acc, pred_number = eval_regression(
            response=response,
            true_answer=float(row_dict["answer"]),
            acceptance_threshold=args.acceptance_threshold
        )
        record = {
            **row_dict,
            "response": response.replace("\n", "").strip(),
            "pred_number": pred_number,
            "sMAPE_acc": sMAPE_acc,
            "correct": correct
        }
    else:
        record = {
            **row_dict,
            "response": response.replace("\n", "").strip(),
            "correct": parse_response(response, row_dict["answer"])
        }
    results.append(record)

    if row_idx % args.save_interval == 0:
        results_df = pd.DataFrame(results)
        if row_idx == 0:
            results_df.to_csv(file_path, index=False, mode="w")
        else:
            results_df.to_csv(file_path, index=False, mode="a", header=False)
        results = list()
if args.verbose:
    tq.close()
if len(results) > 0:
    results_df = pd.DataFrame(results)
    results_df.to_csv(file_path, index=False, mode="a", header=False)
    print(f"Saved results to {file_path}")
    # load entire results file again to compute correct mean
    acc = pd.read_csv(file_path).correct.mean()
    print(f"Accuracy: {acc:.2%}")
else:
    print("No results to save.")

