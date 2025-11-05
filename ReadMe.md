# Gaussian Process Modeling Tutorial

This repository contains a comprehensive tutorial on Gaussian Process (GP) modeling, demonstrating the impact of different normalization methods and kernel choices on model performance.

## Project Structure

```
├── 00_gen_dummy_data.py    # Generate synthetic dataset
├── 01_split_dataset.py     # Split data into train/test sets
├── 02_train_gps.py        # Train GP models with different configurations
├── 03_deploy_model.py     # Deploy the trained GP model
├── 04_visualize.py        # Visualize results and model performance
├── dummy_data.csv         # Generated synthetic dataset
├── data-splits/           # Directory for train/test splits
└── utils/                 # Utility functions
    ├── gp_tools.py       # GP model building and prediction tools
    └── normalize.py      # Data normalization utilities
```

## Features

- Generation of synthetic data (noisy exponentially decaying sine function)
- Implementation of various normalization methods:
  - Standardization
  - Min-Max Scaling
  - Log Standardization
  - Log+b Standardization
  - Square Root Transformation
- Multiple kernel options:
  - RBF (Radial Basis Function)
  - RQ (Rational Quadratic)
  - Matérn 3/2
  - Matérn 5/2
- K-fold cross-validation support
- Visualization tools for model performance analysis (*not completed yet*)

## Getting Started

1. First, generate the synthetic dataset:
   ```bash
   python 00_gen_dummy_data.py
   ```

2. Split the dataset into training and testing sets:
   ```bash
   python 01_split_dataset.py
   ```

3. Train GP models with different configurations:
   ```bash
   python 02_train_gps.py
   ```

4. Deploy the trained model:
   ```bash
   python 03_deploy_model.py
   ```

5. Visualize the results:
   ```bash
   python 04_visualize.py
   ```

## Dependencies

- NumPy
- Pandas
- GPflow
- Scikit-learn
- Matplotlib
- tqdm

To ensure reproducibility, please install the conda environment as such: 

```
conda create -f env.yml
```

## Usage Notes

- The synthetic dataset is generated using an exponentially decaying sine function with added Gaussian noise
- All scripts assume there is in index column at position 0 and that there are no meta-data columns (i.e. all columns other than the target column are features).
- Different normalization methods can significantly impact GP model performance
- The tutorial includes various kernel options to demonstrate their effects on modeling different types of data patterns
- Visualization tools help in understanding the model's performance and the effects of different configurations


