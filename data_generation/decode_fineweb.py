import argparse
import os

import pandas as pd
from tqdm import tqdm

# Set up argument parser
parser = argparse.ArgumentParser(description="Decode parquet files to text files.")
parser.add_argument("--folder_dir", type=str, help="Directory containing parquet files.")
parser.add_argument("--save_path", type=str, help="Directory to save decoded text files.")
args = parser.parse_args()

# Loop through all files in the directory
for file in tqdm(os.listdir(args.folder_dir)):
    if file.endswith(".parquet"):
        path = os.path.join(args.folder_dir, file)
        # read parquet
        df = pd.read_parquet(path)
        text = "<|eos|>\n".join(df["text"].tolist())
        with open(f"{args.save_path}/{file.split("/")[-1].split(".")[0]}.txt", "w") as f:
            f.write(text)
print("Done")
