import argparse
import glob
from pathlib import Path

import pandas as pd
import tqdm
from data_gen_utils import get_strat_params
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description="Collect file paths using a glob pattern.")
parser.add_argument('--globpath', type=str, help='Glob pattern to collect file paths')
parser.add_argument('--num_samples', type=int, default=100_000, help='Number of samples to collect')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
args = parser.parse_args()


file_paths = [Path(f) for f in glob.glob(args.globpath)]
print(file_paths)
tq = tqdm.tqdm(file_paths, desc="file", leave=False)
for filepath in tq:
    tq.set_description(str(filepath))
    params = get_strat_params(filepath.name)
    print(params)

    df = pd.read_csv(filepath, dtype=str, compression="gzip" if filepath.suffix == ".gz" else None)
    
    # Identify groups with fewer than 2 members
    group_counts = df[params].value_counts()
    valid_groups = group_counts[group_counts >= 2].index

    df_valid = df[df[params].apply(tuple, axis=1).isin(valid_groups)]
    df_invalid = df[~df[params].apply(tuple, axis=1).isin(valid_groups)]
    print(f"Invalid groups: {df_invalid.shape[0]}, valid samples: {df_valid.shape[0]}, valid groups: {len(valid_groups)}")
    percentage_num_samples = args.num_samples / df_valid.shape[0]
    num_invalid_test = min(df_invalid.shape[0], int(args.num_samples * (1-percentage_num_samples)))
    df_train_valid, df_test_valid = train_test_split(
        df_valid,
        test_size=args.num_samples -num_invalid_test,
        random_state=args.seed,
        stratify=df_valid[params]
    )
    df_test = pd.concat([df_test_valid, df_invalid.sample(num_invalid_test, random_state=args.seed)])
    assert df_test.shape[0] == args.num_samples, f"Expected {args.num_samples} samples, got {df_test.shape[0]}"
    path = str(filepath).removesuffix(".gz").replace(".csv",f"_{args.num_samples//1_000}k.csv")
    df_test.to_csv(path, index=False)
    print(f"Saved {args.num_samples} samples to {path}")

print("Done")

