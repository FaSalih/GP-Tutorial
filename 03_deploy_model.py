
import pandas as pd
import numpy as np
import os, argparse, pickle

from utils.normalize import normalize
from utils.gp_tools import gpPredict

# Define normalization methods to loop over - complete list = ['Standardization', 'MinMax', 'LogStand', 'Log+bStand', 'Sqrt']
featureNorm = None
labelNormalizations = ['Standardization', 'MinMax', 'Log+bStand', 'Sqrt']

# Kernels to Consider
kernels = ['RBF', 'RQ', 'Matern32', 'Matern52']

def load_gp_model(model_file):
    with open(model_file, "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model

def load_scaler(scaler_file):
    with open(scaler_file, "rb") as f:
        X_train_scaler, Y_train_scaler, featureNorm, labelNorm = pickle.load(f)
    return X_train_scaler, Y_train_scaler, featureNorm, labelNorm

def deploy_models(data_file, n_folds, target_col):
    # Read data for fitting
    df_dat = pd.read_csv(data_file, index_col=0)
    if target_col is not None:
        X_dat = df_dat.drop(columns=[target_col], axis=1)
        # Y_dat = df_dat[target_col]
    else:
        X_dat = df_dat.copy(deep=True)

    # Define  folders for saved trained models and scalers
    models_folder = f"models"
    scalers_folder = f"scalers" 
    metrics_folder = f"metrics" 
    pred_folder = f"predictions" 

    os.makedirs(pred_folder, exist_ok=True)

    # Loop over folds
    for k_fold in range(1,n_folds+1):
        for labelNorm in labelNormalizations:
            for kernel in kernels:
                # Load desired GP model and scaler
                scaler_file = f"{scalers_folder}/{kernel}_{labelNorm}_fold_{k_fold}.pkl"
                model_file = f"{models_folder}/{kernel}_{labelNorm}_fold_{k_fold}.pkl"
                model = load_gp_model(model_file)
                X_scaler, Y_scaler, featureNorm, labelNorm = load_scaler(scaler_file) 

                # Deploy loaded model
                X_dat_norm, _ = normalize(X_dat, skScaler=X_scaler, method=featureNorm)

                Y_pred_norm, Y_std = gpPredict(model,X_dat_norm)

                # Un-normalize predictions
                Y_pred, _ = normalize(Y_pred_norm, skScaler=Y_scaler, method=labelNorm, reverse=True)
                Y_std, _ = normalize(Y_std, skScaler=Y_scaler, method=labelNorm, reverse=True)

                # Save to df
                df_res = df_dat.copy(deep=True)
                df_res[f"{target_col}-Prediction"] = Y_pred
                df_res[f"{target_col}-Prediction STD"] = Y_std

                df_res.to_csv(f'{pred_folder}/{kernel}_{labelNorm}_fold_{k_fold}.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to csv file containing feature data. Column 0 is assumed to be a non-trivial index column. It is assumed that there is no target column, unless a name is provided with the `target` argument.")
    parser.add_argument("--folds", type=int, required=True, help="Number of k-folds or data splits to generate. This argument is required.")
    parser.add_argument("--target", default=None, help="Name of target column to drop from csv to leave only samples index and features. Default is `None`")
    args = parser.parse_args()

    deploy_models(args.csv, args.folds, args.target)
