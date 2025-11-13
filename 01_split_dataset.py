"""
Split dataset into training and testing using using stratified k-folds.
The input csv file, name of target column and number of folds are user-
provided arguments.
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import argparse, os, subprocess
import numpy as np
from utils import visualize as vis

def split_data(csv_path, target_col_name, n_folds, n_bins, out_dir):
    # read the property data
    df = pd.read_csv(csv_path,index_col=0)
    prop = df[target_col_name]

    # Initialize a stratified k-fold object
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Cut the property data into discrete bins
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
    parser.add_argument("--csv", required=True, help="Path to csv file containing target data. Column 0 is assumed to be a non-trivial index column.")
    parser.add_argument("--target", required=True, help="Name of csv column containing target data.")
    parser.add_argument("--dir", default="data-splits", help="Path to directory to save results in. The folder will be created if it does not already exist. Default value is `data-splits`.")
    parser.add_argument("--folds", type=int, default=4, help="Number of k-folds or data splits to generate. Default value is 5.")
    parser.add_argument("--bins", type=int, default=15, help="Number of bins to discretize continuous target property into. Default value is 15.")
    args = parser.parse_args()

    split_data(args.csv, args.target, args.folds, args.bins, args.dir)
    vis.data_splits(args.target, args.folds, args.bins, args.dir)