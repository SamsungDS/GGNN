from ase.symbols import symbols2numbers
import numpy as np
from ase.data import atomic_numbers
from typing import List
import os
import yaml


def append_statistics(statistics, stats):
    if statistics is None:
        statistics = stats
    else:
        for k in statistics:
            if k in "_composition":
                m1 = statistics[k]
                m2 = stats[k]
                statistics[k].update(
                    {
                        "rows": np.concatenate(
                            [m1["rows"], m2["rows"] + m1["shape"][0]]
                        ),
                        "cols": np.concatenate([m1["cols"], m2["cols"]]),
                        "vals": np.concatenate([m1["vals"], m2["vals"]]),
                        "shape": [
                            m1["shape"][0] + m2["shape"][0],
                            max(m1["shape"][1], m2["shape"][1]),
                        ],
                    }
                )
            elif not k.startswith("_"):
                if len(stats[k].keys()) == 0:
                    continue
                n1 = np.prod(statistics[k]["_array_shape"])
                n2 = np.prod(stats[k]["_array_shape"])
                N = n1 + n2
                mean1 = statistics[k]["mean"]
                mean2 = stats[k]["mean"]
                std1 = statistics[k]["std"]
                std2 = stats[k]["std"]
                mean = float((n1 * mean1 + n2 * mean2) / N)
                variance = float(
                    (
                        ((n1 - 1) * std1**2 + (n2 - 1) * std2**2)
                        + n1 * (mean1 - mean) ** 2
                        + n2 * (mean2 - mean) ** 2
                    )
                    / (N - 1)
                )
                std = np.sqrt(variance)

                statistics[k].update(
                    {
                        "mean": mean,
                        "std": std,
                        "median": float(
                            (n1 * statistics[k]["median"] + n2 * stats[k]["median"]) / N
                        ),
                        "max": float(max(statistics[k]["max"], stats[k]["max"])),
                        "min": float(min(statistics[k]["min"], stats[k]["min"])),
                    }
                )
            elif k == "_natoms":
                refkey = "_elemwise_reference_energies"
                elemwise = stats[refkey]
                for sym in elemwise:
                    if sym in statistics[refkey]:
                        statistics[refkey][sym] = (
                            statistics[refkey][sym] * statistics["_natoms"][sym]
                            + elemwise[sym] * stats["_natoms"][sym]
                        ) / (statistics["_natoms"][sym] + stats["_natoms"][sym])
                    else:
                        statistics[refkey][sym] = elemwise[sym]
                for sym, counts in stats[k].items():
                    if sym not in statistics[k]:
                        statistics[k][sym] = 0
                    statistics[k][sym] += counts

    return statistics


def get_type_mapper_from_specie(specie_list: List[str]):
    """
    from ['Hf', 'O']
    return {72: 0, 8: 1}
    """
    specie_list = sorted(specie_list)
    type_map = {}
    unique_counter = 0
    for specie in specie_list:
        atomic_num = symbols2numbers(specie)[0]
        if atomic_num in type_map:
            continue
        type_map[atomic_num] = unique_counter
        unique_counter += 1
    return type_map


def read_statistics(stat_dir, scale_by, shift_by):
    statistics = None
    for filename in stat_dir:
        with open(os.path.join(filename, "stats.yaml")) as f:
            stats = yaml.safe_load(f)
        for k in stats:
            if k.startswith("_"):
                if k == "_composition":
                    array = np.load(
                        os.path.join(stats["_basedir"], stats[k]["_array_path"])
                    )
                    stats[k] = {
                        "rows": array["r"],
                        "cols": array["c"],
                        "vals": array["v"],
                        "shape": stats[k]["_array_shape"],
                    }
                continue
        statistics = append_statistics(statistics, stats)
    chemical_species = [z for z in statistics["_natoms"].keys() if z != "total"]
    if scale_by == "force_rms":
        scale = force_rms(statistics)
    else:
        scale = per_atom_energy_std(statistics)

    if shift_by == "elemwise_reference_energies":
        shift = elemwise_reference_energies(statistics)
    else:
        shift = per_atom_energy_mean(statistics)
    chemical_specie = sorted([x.strip() for x in chemical_species])
    type_map = get_type_mapper_from_specie(chemical_specie)
    return scale, shift, type_map, chemical_specie


def per_atom_energy_mean(statistics):
    return statistics["per_atom_energy"]["mean"]


def elemwise_reference_energies(statistics):
    full_coeff = np.zeros(119)
    for symbol, val in statistics["_elemwise_reference_energies"].items():
        full_coeff[atomic_numbers[symbol]] = val
    _elemwise_reference_energies = full_coeff.tolist()
    return _elemwise_reference_energies


def force_rms(statistics):
    mean = statistics["force_of_atoms"]["mean"]
    std = statistics["force_of_atoms"]["std"]
    return float((mean**2 + std**2) ** 0.5)


def per_atom_energy_std(statistics):
    return statistics["per_atom_energy"]["std"]
