import argparse, os
from utils import visualize as vis

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    vis.data_splits(args.target, args.folds, args.bins, args.dir)
    vis.vis_metrics(metrics_dir="metrics", save_dir="metrics", cmap='Blues')