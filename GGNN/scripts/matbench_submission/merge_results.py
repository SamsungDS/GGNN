import argparse
import os
from glob import glob

import pandas as pd


def main() -> None:
    """Calculate F1 score and RMSD for matbench-discovery predictions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Results directory")
    args = parser.parse_args()
    files = sorted(glob(f"{args.results}/0*_*.json.gz"))
    dataframes = [
        pd.read_json(file_path).set_index("material_id") for file_path in files
    ]
    df_energy = pd.concat(dataframes)

    files = sorted(glob(f"{args.results}/0*_*.jsonl.gz"))
    dataframes = [
        pd.read_json(file_path, lines=True).set_index("material_id")
        for file_path in files
    ]
    df_structure = pd.concat(dataframes)
    df_energy.to_csv(f"{args.results}/energy.csv.gz")
    df_structure.reset_index().to_json(
        f"{args.results}/structure.jsonl.gz",
        orient="records",
        lines=True,
        compression="gzip",
    )


if __name__ == "__main__":
    main()
