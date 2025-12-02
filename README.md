# Physics-constrained High-entropy Optimization and Exploration (PHEONIX) framework

![Workflow Diagram](workflow.png)

## Introduction

**PHEONIX** is a closed-loop high-throughput computational framework designed for the exploration and optimization of High-Entropy Alloys (HEAs) in catalysis.

Navigating the vast compositional and configurational space of HEAs presents a significant challenge for traditional DFT methods. PHEONIX addresses this by integrating thermodynamic rules, Monte Carlo sampling, topological analysis, and geometric deep learning (EquiformerV2) to accelerate the discovery of high-performance catalysts.


## Framework Overview

The PHEONIX framework automates the following key computational stages:

1.  **Thermodynamics-Guided Structure Generation:**
    *   Utilizes **Monte Carlo (MC)** sampling strictly governed by multidimensional thermodynamic constraints ($\Delta S_{mix}$, $\Delta H_{mix}$, $\delta$, and $\Omega$).
    *   Explicitly models lattice distortions and chemical disorder to generate Stable, Unique, and New (SUN) HEA candidates.

2.  **3D Topological Site Enumeration:**
    *   Goes beyond traditional 2D surface models by implementing a topology-based algorithm to identify adsorption sites on both **Surface** (Top, Bridge, Hollow) and **Sub-surface** layers.
    *   Enables the investigation of cross-layer diffusion channels and subsurface mechanisms.

3.  **ML-Accelerated Energy Mapping:**
    *   Integrates the **EquiformerV2** graph neural network to predict adsorption free energies ($\Delta G_{H*}$) with DFT-level accuracy.
    *   Capable of handling massive datasets (demonstrated on >1.16 million sites) to map the complete energy landscape.

4.  **Statistical Descriptor Analysis:**
    *   Implements novel statistical descriptors to quantify energy distributions:
        *   **Energy Overlap Coefficient (EOC):** Measures the thermodynamic coupling between surface and subsurface layers.
        *   **Empirical Band Intersection (EmpBand):** Identifies the active energy window for diffusion.
        *   **Directional Drivability (DD):** Quantifies the thermodynamic driving force for surface-to-subsurface transport.



## Data release


- **v1.1** Release:  
  - [Data.zip (all models and adsorption‐energy statistics)].(https://github.com/QsenQY/HELIOS-Hydrogen-Evolution-via-Learning-Intelligent-Optimization-of-Superalloys-/releases/download/V1.0.2/Data.zip).


## External Dependencies
**fairchem

- **Documentation** (upstream):  
  https://github.com/facebookresearch/fairchem#readme


### Generate HEAs Structues
### `generate_hea.py`
High-throughput generation of HEA slab structures given an element list and cell parameters.

**Usage**  
```bash
# View all available options
python generate_hea.py --help

# Example: generate HEA-1–HEA-5 slabs and save to data/HEA-slabs/
python generate_hea.py \
  --structures fcc,bcc \
  --miller 1 1 1 \
  --layers 4 \
  --vacuum 12.0 \
  --supercell 3 3 \
  --num_alloys 500 \
  --min_elements 5 \
  --max_elements 7 \
  --output ./data/HEA-slabs \
  --enthalpy_file path/to/Mixing_Enthalpy.xlsx
```
**Arguments**  
- `--num_alloys`     : Number of random HEA structures to generate (int, default: 1000)  
- `--min_elements`   : Minimum number of distinct elements per alloy (int, default: 5)  
- `--max_elements`   : Maximum number of distinct elements per alloy (int, default: 7)  
- `--structures`     : Comma-separated list of crystal lattices to sample (`fcc`, `bcc`, `hcp`; default: `"fcc,bcc,hcp"`)  
- `--miller`         : Miller indices for the slab surface cut, three integers (e.g. `1 1 1`; default: `1 1 1`)  
- `--layers`         : Number of atomic layers in each slab (int, default: 4)  
- `--vacuum`         : Vacuum thickness along z-axis in Å (float, default: 10.0)  
- `--supercell`      : In-plane supercell repetition factors a b (two ints, default: `2 3`)  
- `--output`         : Output directory for POSCAR files and CSV summary (string, default: `./HEA_slabs`)  
- `--enthalpy_file`  : Path to Excel file containing the mixing enthalpy matrix (string, **required**)  





### Find adsorption and places adsorbate
### `add_adsorbate.py`
Automatically identifies surface adsorption sites on prebuilt HEA slabs (via convex‐hull screening) and places a specified adsorbate (e.g. H) at each site, writing one VASP POSCAR per site.
**Usage**  
```bash
# View all available options
python add_adsorbate.py --help

# Example: on each .vasp slab in data/HEA-slabs, identify the top-70th-percentile atoms,
# generate Top/Bridge/Hollow sites 1.8 Å above the surface, and write results to data/HEA-adsorbate/
python add_adsorbate.py \
  --input_dir  ../data/HEA-slabs \
  --output_dir ../data/HEA-adsorbate \
  --percentile 70      \
  --ads_dist   1.8     \
  --dist_thr   3.0     \
  --margin     0.2     \
  --adsorbate  H
```
**Arguments**  
- `--input_dir`   : Directory containing slab files (`.vasp` or `.xyz`)  
- `--output_dir`  : Directory to write slab + adsorbate POSCARs  
- `--percentile`  : z‐coordinate screening percentile for candidate surface atoms (default: 70)  
- `--ads_dist`    : Vertical distance above surface to place adsorbate in Å (default: 1.8)  
- `--dist_thr`    : Max interatomic distance in Å for bridge/hollow site detection (default: 3.0)  
- `--margin`      : Minimum height margin above the original surface in Å (default: 0.2)  
- `--adsorbate`   : Adsorbate element symbol (currently supports only `H` or `O`, default: `H`; support for other species coming soon)  

### Quickstart: Batch Prediction with fairchem 

```python
### Quickstart: Batch Prediction with fairchem + BFGS Relaxation

```python
#!/usr/bin/env python3
from fairchem.core import OCPCalculator
from ase.io import read
from ase.optimize import LBFGS
import pandas as pd
import os

# ─── User settings ─────────────────────────────────────────
input_dir  = "data/HEA-adsorbate"          # your folder of .vasp models
output_csv = "results/predictions.csv"     # where to save predictions
fmax       = 0.05                          # force convergence criterion (eV/Å)
maxsteps   = 100                           # max relaxation steps
# ────────────────────────────────────────────────────────────

# 1. Initialize the pretrained EquiformerV2 model
calc = OCPCalculator(
    model_name="EquiformerV2-31M-S2EF-OC20-All+MD",
    local_cache="pretrained_models",
    cpu=False,
)

records = []
for fn in os.listdir(input_dir):
    if not fn.endswith(".vasp"):
        continue

    # 2. Read slab + adsorbate and attach calculator
    atoms = read(os.path.join(input_dir, fn))
    atoms.calc = calc

    # 3. Relax geometry with LBFGS
    dyn = LBFGS(atoms, logfile=None)
    dyn.run(fmax=fmax, steps=maxsteps)

    # 4. Compute adsorption energy
    e_ads = atoms.get_potential_energy()
    records.append({"file": fn, "adsorption_energy_eV": e_ads})
    print(f"{fn}: {e_ads:.3f} eV")

# 5. Save all predictions
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
pd.DataFrame(records).to_csv(output_csv, index=False)
print(f"\nAll predictions saved to {output_csv}")

```
Run:
```bash
python scripts/predict_quick.py
```









