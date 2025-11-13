import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib import rc
import pandas as pd
import os
import glob
from pathlib import Path

# =============================================================================
# Plot Configuration
# =============================================================================

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))
plt.rcParams["font.family"] = "Serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
rc('axes', labelsize='16')
rc('xtick', labelsize='14')
rc('ytick', labelsize='14')
rc('legend', fontsize='12')
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams["savefig.pad_inches"]=0.02

# =============================================================================
# Functions
# =============================================================================

def data_splits(target_col_name, n_folds, n_bins,  out_dir):
    # Initialize a plot to visualize the distribution of the property data
    nrows = 2
    ncols = int(np.ceil(n_folds/nrows))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))
    axs = axs.flatten()
    
    # Loop over the training and testing indices
    for k in range(n_folds):
        # Read the training and testing property data to a csv file
        train_df = pd.read_csv(f"{out_dir}/train_fold_{k+1}.csv", index_col=0)
        test_df = pd.read_csv(f"{out_dir}/test_fold_{k+1}.csv", index_col=0)
        y_train = train_df[target_col_name]
        y_test = test_df[target_col_name]

        # Plot the distribution of the property data
        ax = axs[k]
        ax.hist(y_train, bins=n_bins, alpha=0.5, color='b', label='Training')
        ax.hist(y_test, bins=n_bins, alpha=0.5, color='r', label='Testing')
        ax.set_xlabel(target_col_name)
        ax.set_ylabel('Frequency')
        if k == 0: ax.legend(loc='upper left')
        ax.set_title(f'Fold {k+1}')
    # Add plot attributes for unnormalized property data
    fig.suptitle(f'Target Distribution for Training and Testing Sets', fontweight='bold', fontsize=14)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/PropertyDistribution.png')
    # plt.show()
    plt.close()

def parity_plot():
    # Placeholder for parity plot implementation
    # Implemented as a stub to be expanded later if needed.
    return None


def vis_metrics(metrics_dir=None, save_dir=None, cmap='Blues'):
    """Read metric arrays from the `metrics` directory and create heatmaps.

    The function looks for files named like `R2_train_arr_1.csv`,
    `R2_test_arr_1.csv`, `MAE_train_arr_1.csv`, ... etc. It computes the
    mean and standard deviation across folds for each kernel x scaler cell,
    then plots two heatmaps per metric (train and test) with each cell
    annotated as "mean\n±\nstd".

    Parameters
    ----------
    metrics_dir : str or Path, optional
        Directory containing the metric CSV files. If None, the function
        will look for a `metrics` folder one level above this file (project root).
    save_dir : str or Path, optional
        Directory to write heatmap images. If None, images are saved in `metrics_dir`.
    cmap : str, optional
        Matplotlib colormap to use for the heatmaps.
    """


    base = Path(__file__).resolve().parent.parent
    metrics_path = Path(metrics_dir) if metrics_dir else base / 'metrics'
    if save_dir:
        save_path = Path(save_dir)
    else:
        save_path = metrics_path
    save_path.mkdir(parents=True, exist_ok=True)

    metrics = ['R2', 'MAE']
    splits = ['train', 'test']

    # Default labels (common ordering used in this project)
    default_kernels = ['RBF', 'RQ', 'Matern32', 'Matern52']
    default_scalers = ['Standardization', 'MinMax', 'LogStand', 'Log+bStand', 'Sqrt']

    for metric in metrics:
        # Read mean arrays for both splits first to get consistent color scales
        mean_arrays = {}
        std_arrays = {}
        found_any = False
        for split in splits:
            pattern = str(metrics_path / f"{metric}_{split}_arr_*.csv")
            files = sorted(glob.glob(pattern))
            if len(files) == 0:
                # No files for this split/metric
                print(f"No files found for pattern: {pattern}")
                continue
            found_any = True
            arrs = []
            for f in files:
                try:
                    df = pd.read_csv(f, header=None, index_col=None)
                    arrs.append(df.values)
                except Exception as e:
                    print(f"Failed reading {f}: {e}")
            if len(arrs) == 0:
                continue
            # Stack along new axis (folds) and compute mean/std across folds
            stack = np.stack(arrs, axis=0)
            mean_arr = np.mean(stack, axis=0)
            std_arr = np.std(stack, axis=0)
            mean_arrays[split] = mean_arr
            std_arrays[split] = std_arr

        if not found_any:
            print(f"No metric files found for metric '{metric}' in {metrics_path}")
            continue

        # Determine a consistent vmin/vmax across train/test for this metric
        all_means = np.hstack([mean_arrays[s].ravel() for s in mean_arrays.keys()])
        vmin = np.nanmin(all_means)
        vmax = np.nanmax(all_means)
        # Avoid zero-range
        if np.isclose(vmin, vmax):
            vmin -= 1e-8
            vmax += 1e-8

        # Choose labels size based on data shape; fallback to defaults if sizes don't match
        # We'll take shape from one of the available mean arrays
        sample_mean = next(iter(mean_arrays.values()))
        n_rows, n_cols = sample_mean.shape
        if n_rows == len(default_kernels):
            kernels = default_kernels
        else:
            kernels = [f'kernel_{i}' for i in range(n_rows)]
        if n_cols == len(default_scalers):
            scalers = default_scalers
        else:
            scalers = [f'scaler_{j}' for j in range(n_cols)]

        # Plot heatmaps for each available split
        for split in mean_arrays.keys():
            mean_arr = mean_arrays[split]
            std_arr = std_arrays[split]

            fig, ax = plt.subplots(figsize=(1.6 * n_cols + 1.5, 1.2 * n_rows + 1.5))
            im = ax.imshow(mean_arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.tick_params(labelsize=10)

            # ticks
            ax.set_xticks(np.arange(n_cols))
            ax.set_yticks(np.arange(n_rows))
            ax.set_xticklabels(scalers, rotation=45, ha='right')
            ax.set_yticklabels(kernels)

            # Annotate each cell with mean ± std in two lines
            for i in range(n_rows):
                for j in range(n_cols):
                    m = mean_arr[i, j]
                    s = std_arr[i, j]
                    # Format numbers: show two decimals for readability
                    text = f"{m:.2f}\n±\n{s:.2f}"
                    # Choose text color based on normalized mean for contrast
                    norm_val = (m - vmin) / (vmax - vmin) if (vmax - vmin) != 0 else 0
                    text_color = 'white' if norm_val > 0.55 else 'black'
                    ax.text(j, i, text, ha='center', va='center', color=text_color, fontsize=10)

            ax.set_title(f"{metric} — {split.capitalize()} dataset", fontsize=14, fontweight='bold')
            fig.tight_layout()

            out_file = save_path / f"{metric}_{split}_heatmap.png"
            fig.savefig(out_file, dpi=300)
            plt.close(fig)
            print(f"Saved heatmap: {out_file}")

    return None
    