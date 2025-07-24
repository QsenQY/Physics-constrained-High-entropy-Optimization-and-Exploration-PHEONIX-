# HELIOS Hydrogen-Evolution-via-Learning-Intelligent-Optimization-of-Superalloys
Machine learning drives automated modeling of high-entropy alloys and prediction of hydrogen production performance

![Workflow Diagram](workflow.png)

## Abstract
High-entropy alloy materials demonstrate exceptional catalytic properties due to their distinctive multi-component attributes and electronic effects. Nonetheless, the extensive data landscape of high-entropy alloys presents substantial hurdles in identifying high-performance catalysts. Compounding this challenge is the anisotropic character of surface sites within each catalyst, rendering performance prediction exceedingly difficult with conventional computational techniques. Although contemporary machine learning models have achieved significant advances in predicting properties for specific principal element combinations of high-entropy alloys, they encounter difficulties when modeling random combinations of principal elements and their concentrations. In this context, we introduce a closed-loop high-throughput workflow that integrates high-throughput automated modeling and performance prediction of high-entropy alloys and adsorption structures, along with a high-throughput synthesis method leveraging microchannel technology. We apply a previously established thermodynamic model to forecast the stability of principal element combinations and harness the MatterGen high-throughput modeling approach to screen 133,233 thermodynamically stable high-entropy alloy materials. Utilizing the high-throughput automated workflow, we have generated over 1,160,000 hydrogen atom adsorption models and, for the first time, have successfully predicted hydrogen adsorption energies using the EquiformerV2 model, pinpointing five high-entropy alloy materials with superior catalytic performance. These materials have been successfully synthesized via microchannel technology, and electrochemical experiments have confirmed their outstanding catalytic properties (overpotential = 5.5–9 mV). Statistical analysis indicates that the performance of active sites on high-entropy alloy surfaces adheres to a bimodal distribution. The unique electronic and structural synergy in high-entropy alloys substantially diminishes the inherent properties of principal elements, leading to localized averaging effects. This research presents innovative concepts and methodologies for navigating the data space of high-entropy alloy materials and for high-throughput prediction of catalytic performance.


## Data release


- **v1.1** Release:  
  - [Data.zip (all models and adsorption‐energy statistics)].(https://github.com/QsenQY/HELIOS-Hydrogen-Evolution-via-Learning-Intelligent-Optimization-of-Superalloys-/releases/download/V1.0.2/Data.zip).

## External Dependencies

This project uses Git submodules to integrate two external repositories. Before proceeding, clone and initialize all submodules:

```bash
git clone --recurse-submodules git@github.com:QsenQY/HELIOS-Hydrogen-Evolution-via-Learning-Intelligent-Optimization-of-Superalloys-.git
cd HELIOS-Hydrogen-Evolution-via-Learning-Intelligent-Optimization-of-Superalloys-
git submodule update --init --recursive
```

### fairchem

- **Documentation** (upstream):  
  https://github.com/facebookresearch/fairchem#readme

- **Source (pinned)**:  
  https://github.com/QsenQY/fairchem/tree/977a80328f2be44649b414a9907a1d6ef2f81e95


### mattergen

- **Documentation** (upstream):  
  https://github.com/microsoft/mattergen#readme

- **Source (pinned)**:  
  https://github.com/QsenQY/mattergen/tree/ec029d177c93709fa9a2ea4e48b872760d09c63b

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









