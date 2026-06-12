# GGNN

GGNN is a Geometric GNN-based MLFF (Machine Learning Force Field) package that supports E(3)-equivariant models built on Clebsch–Gordan tensor products through e3nn, with accelerated tensor product computation via cuequivariance and FlashTP. The package includes the EquFlash and EquFlashV2 MLFF architectures.

The core structure of the package is based on [fairchem-core](https://github.com/FAIR-chem/fairchem).

## Install

```bash
pip install -r requirements.txt
pip install --no-deps -r requirements-no-deps.txt
```

**Requirements:** Python 3.12, PyTorch 2.9.1 (CUDA 12.6)

## Training

```bash
# Start new training
python main.py --mode train --config-yml configs/*.yml

# Resume from checkpoint
python main.py --mode train --config-yml configs/*.yml --checkpoint checkpoint.pt
```

## Models

| `model.name` | Description |
|---|---|
| `equflash` | Base Equflash model |
| `equflash_comp` | TorchScript-compiled Equflash |
| `equflash_pol` | E/F + Polarization and Born charge prediction |
| `equflash_bec` | E/F + Born effective charge prediction |
| `equflash_field` | E/F + Polarization prediction via electric field response|
| `equflashv2` | Base EquflashV2 model |
| `equflashv2_comp` | TorchScript-compiled EquflashV2 |

## Checkpoints

| Version | Download |
|---|---|
| `equflash` | [figshare](https://figshare.com/ndownloader/files/65435004) |
| `equflashv2` | [figshare](https://figshare.com/ndownloader/files/65435007) |

## Calculator

```python
from GGNN.common.calculator import UCalculator

atoms.calc = UCalculator(checkpoint_path="checkpoint.pt", cpu=False)
```

## Evaluation

```bash
# MatBench (f1 + rmsd)
bash scripts/submit_matbench.sh <checkpoint> <worldsize>   # default worldsize=8
python -m GGNN.scripts.merge_f1_and_rmsd --results <checkpoint_dir>/matbench_results

# KSRME thermal conductivity
bash scripts/run_ksrme.sh <checkpoint> <output_dir>
```

---

## MD Simulation Guides

Below are two ways to run molecular dynamics with the released GGNN checkpoint:

1. **TorchSim** — recommended for batched / GPU-accelerated MD on multiple systems.
2. **ASE** — quick single-system runs and integration with the wider ASE ecosystem.

Both guides assume you have an installed `GGNN` package and a model checkpoint
(e.g. `checkpoint.pt`).

---

## 1. TorchSim MD Guide

**Tested with TorchSim v0.6.0.** The wrapper and examples below target this
exact release — newer versions may change the public API.

[TorchSim](https://github.com/TorchSim/torch-sim) (v0.6.0) is a PyTorch-native
atomistic simulation engine that exposes a `ModelInterface`. We ship a
ready-made wrapper, `GGNN.common.torchsim_model.GgnnModel`, that adapts a
trained GGNN checkpoint to that interface so you can use TorchSim's
high-level `integrate` / `optimize` / `static` runners as well as the
low-level integrator stepping API.

### Install

```bash
pip install torch-sim-atomistic==0.6.0
```

(plus the GGNN dependencies — see the project root `requirements.txt`.)

### High-level API: batched NVT MD on many systems

```python
import torch
import torch_sim as ts
from ase.build import bulk

from GGNN.common.torchsim_model import GgnnModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wrap a GGNN checkpoint as a TorchSim model
model = GgnnModel(
    model="checkpoint.pt",        # path to the released checkpoint
    device=device,
    dtype=torch.float32,
    compute_forces=True,
    compute_stress=True,
)

# Build a list of ASE Atoms — TorchSim batches them automatically
cu = bulk("Cu", "fcc", a=3.58, cubic=True).repeat((2, 2, 2))
systems = [cu] * 16
trajectory_files = [f"Cu_traj_{i}.h5md" for i in range(len(systems))]

final_state = ts.integrate(
    system=systems,
    model=model,
    integrator=ts.Integrator.nvt_langevin,
    n_steps=1000,
    timestep=0.002,                # ps
    temperature=1000,              # K
    trajectory_reporter=dict(
        filenames=trajectory_files,
        state_frequency=10,
    ),
)

# Convert the final batched state back into a list of ASE Atoms
final_atoms = final_state.to_atoms()

# Pull final potential energies from the trajectory files
for fname in trajectory_files:
    with ts.TorchSimTrajectory(fname) as traj:
        print(fname, traj.get_array("potential_energy")[-1])
```

`ts.integrate` accepts a single `Atoms`, a list of `Atoms`, or a `pymatgen`
`Structure`. Set `autobatcher=True` to let TorchSim pick the optimal batch
size for your GPU.

### Choosing an ensemble

Pick the integrator via the `ts.Integrator` enum:

| Ensemble | `ts.Integrator` value |
|---|---|
| NVE (microcanonical) | `nve` |
| NVT — velocity rescaling | `nvt_vrescale` |
| NVT — Langevin (BAOAB) | `nvt_langevin` |
| NVT — Nosé–Hoover | `nvt_nose_hoover` |
| NPT — Langevin isotropic | `npt_langevin_isotropic` |
| NPT — Langevin anisotropic | `npt_langevin_anisotropic` |
| NPT — Nosé–Hoover isotropic | `npt_nose_hoover_isotropic` |
| NPT — C-rescale isotropic | `npt_crescale_isotropic` |
| NPT — C-rescale triclinic | `npt_crescale_triclinic` |

For NPT runs make sure the model is constructed with `compute_stress=True`.

### Low-level API: explicit step loop

If you want full control (custom logging, on-the-fly analysis, mixed
ensembles), drive the step functions directly:

```python
import torch
import torch_sim as ts
from ase.build import bulk
from torch_sim.units import MetalUnits as Units

from GGNN.common.torchsim_model import GgnnModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

model = GgnnModel(
    model="checkpoint.pt",
    device=device,
    dtype=dtype,
)

si = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
state = ts.io.atoms_to_state(si, device=device, dtype=dtype)

dt = torch.tensor(0.002 * Units.time, device=device, dtype=dtype)   # 2 fs
kT = torch.tensor(1000 * Units.temperature, device=device, dtype=dtype)
gamma = torch.tensor(10 / Units.time, device=device, dtype=dtype)

state = ts.nvt_langevin_init(state=state, model=model, kT=kT)
for step in range(2000):
    state = ts.nvt_langevin_step(
        state=state, model=model, dt=dt, kT=kT, gamma=gamma
    )
    if step % 100 == 0:
        T = ts.calc_kT(
            masses=state.masses,
            momenta=state.momenta,
            system_idx=state.system_idx,
        ) / Units.temperature
        print(f"step {step}: T = {T.item():.2f} K")
```

For NPT replace the init/step calls with e.g.
`ts.npt_nose_hoover_isotropic_init` / `ts.npt_nose_hoover_isotropic_step`
and pass an `external_pressure` tensor.

### Geometry optimization (FIRE)

```python
relaxed = ts.optimize(
    system=systems,
    model=model,
    optimizer=ts.Optimizer.fire,
    init_kwargs=dict(cell_filter=ts.CellFilter.frechet),
    convergence_fn=ts.generate_force_convergence_fn(force_tol=1e-3),
)
print(relaxed.energy)
```

---

## 2. ASE MD Guide

For single-system runs or when you need ASE-specific tooling (constraints,
NEB, Phonopy interfaces, dump formats, …), use the bundled ASE calculator
`GGNN.common.calculator.UCalculator`.

### NVT Langevin example

```python
from ase.build import bulk
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import Trajectory
from ase import units

from GGNN.common.calculator import UCalculator

# Attach the GGNN model as an ASE calculator
atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
atoms.calc = UCalculator(
    checkpoint_path="checkpoint.pt",
    cpu=False,          # set True to force CPU
)

# Initial Maxwell–Boltzmann velocities at 2 * T_target (ASE convention)
MaxwellBoltzmannDistribution(atoms, temperature_K=1000)

dyn = Langevin(
    atoms,
    timestep=2.0 * units.fs,
    temperature_K=1000,
    friction=0.01 / units.fs,
)

# Log thermodynamics every 10 steps and write a trajectory
traj = Trajectory("si_nvt.traj", "w", atoms)
dyn.attach(traj.write, interval=10)

def log():
    epot = atoms.get_potential_energy()
    ekin = atoms.get_kinetic_energy()
    T = ekin / (1.5 * units.kB * len(atoms))
    print(f"Epot={epot:.4f} eV  Ekin={ekin:.4f} eV  T={T:.1f} K")

dyn.attach(log, interval=100)
dyn.run(2000)
```

### Other ensembles

ASE provides drop-in alternatives — just swap the dynamics class:

| Ensemble | ASE class |
|---|---|
| NVE (Velocity Verlet) | `ase.md.verlet.VelocityVerlet` |
| NVT Langevin | `ase.md.langevin.Langevin` |
| NVT Nosé–Hoover (NVT-Berendsen / NVT-NoseHooverChain) | `ase.md.nose_hoover_chain.NoseHooverChainNVT` |
| NPT (Parrinello–Rahman + Nosé–Hoover) | `ase.md.npt.NPT` |

For NPT you must enable stress on the calculator side; `UCalculator` will
return stress automatically when the underlying model was trained with
stress targets.

### Geometry optimization

```python
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter

atoms.calc = UCalculator(checkpoint_path="checkpoint.pt", cpu=False)
opt = FIRE(FrechetCellFilter(atoms))
opt.run(fmax=1e-3)
```

---

### When to use which

- Use **TorchSim** when you want to run **many systems in parallel**, take
  advantage of GPU batching, or plug into the autobatcher / binary
  trajectory format.
- Use **ASE** for **single-system** runs, complex workflows that already
  rely on ASE objects (NEB, phonons, constraints), or when you want the
  full ASE I/O and analysis ecosystem.

