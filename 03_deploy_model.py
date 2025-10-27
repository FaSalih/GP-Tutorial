
import pandas as pd
import gpflow, numpy as np, os, argparse, pickle

from utils.normalize import normalize
from utils.gp_tools import buildGP, gpPredict

# Define normalization methods
featureNorm = None
labelNormalizations = ['Standardization', 'MinMax', 'LogStand', 'Log+bStand', 'Sqrt']

# Kernels to Consider
kernels = ['RBF', 'RQ', 'Matern32', 'Matern52']

# Number of Folds
nfolds = 5

def load_gp_model(model_file):
    with open(model_file, "rb") as f:
        loaded_model = pickle.load(f)
    return loaded_model

def load_scaler(scaler_file):
    with open(scaler_file, "rb") as f:
        X_train_scaler, Y_train_scaler, featureNorm, labelNorm = pickle.load(f)
    return X_train_scaler, Y_train_scaler, featureNorm, labelNorm

def deploy_models(data_file, target_col):
    # Read data for fitting
    df_dat = pd.read_csv(data_file, index_col=0)
    X_dat = df_dat.drop(columns=[target_col], axis=1)
    Y_dat = df_dat[target_col]

    # Define  folders for saved trained models and scalers
    models_folder = f"models"
    scalers_folder = f"scalers" 
    metrics_folder = f"metrics" 
    pred_folder = f"predictions" 

    # Loop over folds
    for k_fold in range(1,nfolds+1):
        for labelNorm in labelNormalizations:
            for kernel in kernels:
                # Load desired GP model and scaler
                scaler_file = f"{scalers_folder}/{kernel}_{labelNorm}_fold_{k_fold}.pkl"
                model_file = f"{models_folder}/{kernel}_{labelNorm}_fold_{k_fold}.pkl"
                model = load_gp_model(model_file)
                X_scaler, Y_scaler, featureNorm, labelNorm = load_scaler(scaler_file) 

                # Deploy loaded model
                X_dat_norm = normalize(X_dat, skScaler=X_scaler, method=featureNorm)
                Y_dat_norm = normalize(Y_dat, skScaler=Y_scaler, method=labelNorm)

                Y_pred_norm, Y_std = gpPredict(model,X_dat_norm)

                # Un-normalize predictions
                Y_pred = normalize(Y_pred_norm, skScaler=Y_scaler, method=labelNorm, reverse=True)
                Y_std = normalize(Y_std, skScaler=Y_scaler, method=labelNorm, reverse=True)

                # Save to df
                df_res = pd.copy(df_dat)
                df_res[f"{target_col}-Prediction"] = Y_pred
                df_res[f"{target_col}-Prediction STD"] = Y_std

                df_res.to_csv(f'{pred_folder}/{kernel}_{labelNorm}_fold_{k_fold}.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    deploy_models(args.csv)
