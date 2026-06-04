# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

import argparse
import pathlib

import torch
import logging

from GGNN.common.lammps_mliap.lmp_mliap_wrapper import GGNNLAMMPSMLIAPWrapper





def main():
    # === parse inputs ===
    parser = argparse.ArgumentParser(
        description="Create GGNN LAMMPS ML-IAP file from saved models.",
    )

    # positional arguments:
    parser.add_argument(
        "model_path",
        help="path to a checkpoint model or packaged model file",
        type=pathlib.Path,
    )

    parser.add_argument(
        "output_path",
        help="absolute path to write GGNN LAMMPS ML-IAP interface file",
        type=pathlib.Path,
    )
    args=parser.parse_args()
    logging.info(f"LAMMPS ML-IAP artefact saved to {args.output_path}")
    # === create and save ML-IAP module ===
    mliap_module = GGNNLAMMPSMLIAPWrapper(
        ckpt_path=str(args.model_path),
    )
    torch.save(mliap_module, args.output_path)
    logging.info(f"LAMMPS ML-IAP artefact saved to {args.output_path}")


if __name__ == "__main__":
    main()
