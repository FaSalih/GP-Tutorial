"""
Split dataset into training and testing using using stratified k-folds.
The input csv file, name of target column and number of folds are user-
provided arguments.
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import argparse, os, subprocess
import numpy as np

def split_data(csv_path, target_col_name, n_splits=5, out_dir="data-splits"):
    # read the property data
    df = pd.read_csv(csv_path,index_col=0)
    prop = df[target_col_name]

    # Initialize a stratified k-fold object
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Cut the property data into discrete bins
    n_bins = 15
    prop_cut = pd.cut(prop, bins=n_bins, labels=range(n_bins))

    # Create a placeholder input array
    X = df.drop(columns=[target_col_name], axis=1)
    
    # create and empty directory
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(f"rm {out_dir}/*.csv", shell=True)

    # Loop over the training and testing indices
    for k, (train_idx, test_idx) in enumerate(skf.split(X, prop_cut)):
    
        # Save the training and testing property data to a csv file
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        train_df.to_csv(f"{out_dir}/train_fold_{k+1}.csv", index=True)
        test_df.to_csv(f"{out_dir}/test_fold_{k+1}.csv", index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    split_data(args.csv, args.target, args.folds)