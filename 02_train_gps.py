import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import os, argparse, warnings, pickle
from tqdm import tqdm

from utils.normalize import normalize
from utils.gp_tools import buildGP, gpPredict

# Define normalization methods to loop over - complete list = ['Standardization', 'MinMax', 'LogStand', 'Log+bStand', 'Sqrt']
featureNorm = None
labelNormalizations = ['Standardization', 'MinMax', 'Log+bStand', 'Sqrt']

# Kernels to Consider
kernels = ['RBF', 'RQ', 'Matern32', 'Matern52']

def train_gp_models(k_fold, target_col, out_dir):
    # initialize dataframe to store metrics
    R2_train_df = pd.DataFrame(index=kernels, columns=[])
    mae_train_df = pd.DataFrame(index=kernels, columns=[])
    R2_test_df = pd.DataFrame(index=kernels, columns=[])
    mae_test_df = pd.DataFrame(index=kernels, columns=[])

    # Read training data for current fold
    df_train = pd.read_csv(f"{out_dir}/train_fold_{k_fold}.csv", index_col=0)
    X_train = df_train.drop(columns=[target_col], axis=1)
    Y_train = df_train[target_col]
    # Read testing data for current fold
    df_test = pd.read_csv(f"{out_dir}/test_fold_{k_fold}.csv", index_col=0)
    X_test = df_test.drop(columns=[target_col], axis=1)
    Y_test = df_test[target_col]

    # Define and create folders to save trained models and scalers
    models_folder = f"models"
    scalers_folder = f"scalers" 
    metrics_folder = f"metrics" 
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(scalers_folder, exist_ok=True)
    os.makedirs(metrics_folder, exist_ok=True)

    results = []
    # Loop over kernels
    for ker, kernel in tqdm(enumerate(kernels), desc="Kernels"):
        # Loop over target normalization methods
        for ln, labelNorm in tqdm(enumerate(labelNormalizations), desc="\tNormalizations"):
            # Normalize training data
            X_train_norm, X_scaler = normalize(X_train, method=featureNorm) 
            Y_train_norm, Y_scaler = normalize(Y_train, method=labelNorm) 

            # Normalize testing data
            X_test_norm, _ = normalize(X_test, method=featureNorm, skScaler=X_scaler) 
            Y_test_norm, _ = normalize(Y_test, method=labelNorm,skScaler=Y_scaler ) 

            # GP Configuration
            gpConfig={'kernel':kernel,
                    'useWhiteKernel':False,
                    'trainLikelihood':True,
                    'alpha':10**-2}
            
            # Fit GP model
            model = buildGP(X_train_norm, Y_train_norm, gpConfig=gpConfig)

            # Get GP predictions
            Y_train_pred_norm, STD_Train = gpPredict(model,X_train_norm)
            Y_test_pred_norm, STD_Test = gpPredict(model,X_test_norm)

            # Unnormalize
            if labelNorm is not None:
                Y_train_pred, __ = normalize(Y_train_pred_norm, skScaler=Y_scaler, method=labelNorm, reverse=True)
                Y_test_pred, __ = normalize(Y_test_pred_norm, skScaler=Y_scaler, method=labelNorm, reverse=True)
            else:
                Y_train_pred = Y_train_pred_norm
                Y_test_pred = Y_test_pred_norm

            # Compute metrics
            R2_train = r2_score(Y_train,Y_train_pred)
            R2_test = r2_score(Y_test,Y_test_pred)
            MAE_train = mean_absolute_error(Y_train,Y_train_pred)
            MAE_test = mean_absolute_error(Y_train,Y_train_pred)

            # Save metrics to dataframe
            R2_train_df.loc[kernel, labelNorm] = R2_train
            R2_test_df.loc[kernel, labelNorm] = R2_test
            mae_train_df.loc[kernel, labelNorm] = MAE_train
            mae_test_df.loc[kernel, labelNorm] = MAE_test

            # Save metrics to csv - incremental
            R2_train_df.to_csv(f'{metrics_folder}/R2_train_arr_{k_fold}.csv')
            mae_train_df.to_csv(f'{metrics_folder}/MAE_train_arr_{k_fold}.csv')
            R2_test_df.to_csv(f'{metrics_folder}/R2_test_arr_{k_fold}.csv')
            mae_test_df.to_csv(f'{metrics_folder}/MAE_test_arr_{k_fold}.csv')

            # Save scaler object and method names
            scaler_file = f"{scalers_folder}/{kernel}_{labelNorm}_fold_{k_fold}.pkl"
            with open(scaler_file, "wb") as f: 
                if featureNorm is None: 
                    X_scaler = None 
                if labelNorm is None: 
                    Y_scaler = None 
                pickle.dump((X_scaler, Y_scaler, featureNorm, labelNorm), f)
            
            # Save GP model
            model_file = f"{models_folder}/{kernel}_{labelNorm}_fold_{k_fold}.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(model, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Name of csv column containing target data.")
    parser.add_argument("--kfold", required=True, help="Number of k-fold to run GP-fitting on. If `01_split_datasets.py` was used to generate the data splits, fold numbers start from 1.")
    parser.add_argument("--dir", default="data-splits", help="Path to directory to save results in. The folder will be created if it does not already exist. Default value is `data-splits`.")
    args = parser.parse_args()

    train_gp_models(args.kfold, args.target, args.dir)


