#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flexible HEA Structure Generator (fixed compositions or random sampling)

Key features:
- No hard binding between element and structure. If a per-structure lattice constant
  is missing, estimate it from metallic radius:
    * fcc: a ≈ 2*sqrt(2)*r
    * bcc: a ≈ 4*r/sqrt(3)
    * hcp: a ≈ 2*r, c/a ≈ 1.633 (unless element-specific c is known)
- Two modes:
  (A) Fixed elements + fixed composition (skip Monte Carlo screening)
  (B) Random Monte Carlo sampling with thermodynamic/size-mismatch screening
- Optional enthalpy matrix (Excel) for H_mix and Omega; if absent, skip Omega.
- Lattice constant override options for total cell (final a, c) if you need.
- Clear warnings instead of hard KeyErrors for missing data.

Outputs:
- VASP POSCAR files (.vasp) and a 'alloy_parameters.csv' summary.

Usage example (your case):
python generate_hea_flexible.py \
  --num_alloys 1 \
  --structure fcc \
  --miller 1 1 1 \
  --layers 4 \
  --vacuum 10 \
  --supercell 3 3 \
  --output ./PtFeCoNiCu_HEA \
  --fixed-elements Pt,Fe,Co,Ni,Cu \
  --fixed-composition 0.40,0.1625,0.125,0.1625,0.15
"""

import os
import re
import math
import random
import argparse
import warnings
import numpy as np
from typing import Dict, Tuple, Optional

try:
    import pandas as pd
except Exception:
    pd = None  # only needed if enthalpy file is provided

from ase.build import surface
from ase.io.vasp import write_vasp
from ase.lattice.cubic import FaceCenteredCubic, BodyCenteredCubic
from ase.lattice.hexagonal import HexagonalClosedPacked

R_GAS = 8.314  # J/mol/K

# -------- Periodic "base" data: metallic radius (nm? Å?) -> here use Å for r  --------
# r below are metallic/atomic radii in Å (close enough for lattice estimation).
# Tm in K. Lattice constants are provided if commonly used; otherwise we estimate.
PERIODIC: Dict[str, Dict] = {
    # fcc-like classics
    "Ni": {"r": 1.24, "Tm": 1728, "latt": {"fcc": {"a": 3.52}}},
    "Cu": {"r": 1.28, "Tm": 1358, "latt": {"fcc": {"a": 3.61}}},
    "Pd": {"r": 1.37, "Tm": 1828, "latt": {"fcc": {"a": 3.89}}},
    "Ag": {"r": 1.44, "Tm": 1235, "latt": {"fcc": {"a": 4.09}}},
    "Ir": {"r": 1.36, "Tm": 2739, "latt": {"fcc": {"a": 3.84}}},
    "Pt": {"r": 1.39, "Tm": 2041, "latt": {"fcc": {"a": 3.92}}},
    "Au": {"r": 1.44, "Tm": 1337, "latt": {"fcc": {"a": 4.08}}},
    "Rh": {"r": 1.34, "Tm": 2237, "latt": {"fcc": {"a": 3.80}}},
    # bcc-like
    "V":  {"r": 1.34, "Tm": 2183, "latt": {"bcc": {"a": 3.03}}},
    "Cr": {"r": 1.28, "Tm": 2180, "latt": {"bcc": {"a": 2.88}}},
    "Fe": {"r": 1.26, "Tm": 1811, "latt": {"bcc": {"a": 2.87}, "fcc": {"a": 3.65}}},  # fcc Fe ~ metastable
    "Nb": {"r": 1.46, "Tm": 2750, "latt": {"bcc": {"a": 3.30}}},
    "Mo": {"r": 1.39, "Tm": 2896, "latt": {"bcc": {"a": 3.15}}},
    "Ta": {"r": 1.46, "Tm": 3290, "latt": {"bcc": {"a": 3.31}}},
    "W":  {"r": 1.39, "Tm": 3695, "latt": {"bcc": {"a": 3.16}}},
    # hcp-like
    "Sc": {"r": 1.62, "Tm": 1814, "latt": {"hcp": {"a": 3.31, "c": 5.27}}},
    "Ti": {"r": 1.47, "Tm": 1941, "latt": {"hcp": {"a": 2.95, "c": 4.68}}},
    "Co": {"r": 1.25, "Tm": 1768, "latt": {"hcp": {"a": 2.51, "c": 4.07}, "fcc": {"a": 3.54}}},
    "Zn": {"r": 1.33, "Tm": 693,  "latt": {"hcp": {"a": 2.66, "c": 4.95}}},
    "Zr": {"r": 1.60, "Tm": 2128, "latt": {"hcp": {"a": 3.23, "c": 5.15}}},
    "Ru": {"r": 1.34, "Tm": 2607, "latt": {"hcp": {"a": 2.71, "c": 4.28}}},
    "Cd": {"r": 1.48, "Tm": 594,  "latt": {"hcp": {"a": 2.98, "c": 5.62}}},
    "Hf": {"r": 1.59, "Tm": 2506, "latt": {"hcp": {"a": 3.19, "c": 5.05}}},
    "Re": {"r": 1.37, "Tm": 3459, "latt": {"hcp": {"a": 2.76, "c": 4.46}}},
    "Os": {"r": 1.35, "Tm": 3306, "latt": {"hcp": {"a": 2.74, "c": 4.32}}},
}

DEFAULT_HCP_COA = 1.633

# ---------- Utility functions ----------
def parse_list_str(s: str) -> list:
    return [x.strip() for x in s.split(",") if x.strip()]

def normalize_comp(vec: np.ndarray) -> np.ndarray:
    s = float(np.sum(vec))
    if s <= 0:
        raise ValueError("Composition sums to 0 or negative.")
    if abs(s - 1.0) > 1e-6:
        warnings.warn(f"Compositions sum to {s:.6f}, auto-renormalizing to 1.0.")
    return vec / s

def estimate_lattice_from_radius(element: str, structure: str) -> Tuple[float, Optional[float]]:
    """Estimate per-element (a, c) for the given structure from metallic radius."""
    if element not in PERIODIC or "r" not in PERIODIC[element]:
        raise KeyError(f"Missing metallic radius for element '{element}'. Add it to PERIODIC.")
    r = PERIODIC[element]["r"]
    if structure == "fcc":
        # nearest-neighbor d_nn = a/sqrt(2), r ≈ d_nn/2 => a ≈ 2*sqrt(2)*r
        a = 2.0 * math.sqrt(2.0) * r
        return a, None
    elif structure == "bcc":
        # d_nn = sqrt(3)*a/2, r ≈ d_nn/2 => a ≈ 4*r/sqrt(3)
        a = 4.0 * r / math.sqrt(3.0)
        return a, None
    elif structure == "hcp":
        a = 2.0 * r
        c = DEFAULT_HCP_COA * a
        return a, c
    else:
        raise ValueError("structure must be one of fcc/bcc/hcp")

def get_element_lattice(element: str, structure: str) -> Tuple[float, Optional[float], bool]:
    """
    Return (a, c, is_estimated).
    If not found in table, estimate from metallic radius.
    """
    tbl = PERIODIC.get(element, {})
    latt = tbl.get("latt", {}).get(structure, {})
    a = latt.get("a", None)
    c = latt.get("c", None)
    if a is None:
        a, c = estimate_lattice_from_radius(element, structure)
        return a, c, True
    else:
        # c may still be None (for fcc/bcc) — that's fine
        return a, c, False

def average_lattice_constants(elements, proportions, structure) -> Tuple[float, Optional[float]]:
    """
    Weighted average of per-element lattice constants.
    Missing per-element constants are estimated; warns once for each estimated element.
    """
    a_list, c_list, p_list = [], [], []
    warned = set()
    for el, p in zip(elements, proportions):
        a_i, c_i, est = get_element_lattice(el, structure)
        if est and el not in warned:
            warnings.warn(f"[Lattice] '{el}' in {structure} not tabulated — estimated from radius.")
            warned.add(el)
        a_list.append(a_i)
        c_list.append(c_i if c_i is not None else 0.0)
        p_list.append(p)

    p_arr = np.array(p_list, float)
    a_avg = float(np.dot(a_list, p_arr) / p_arr.sum())
    # if any c_i existed, do a weighted mean; otherwise None for fcc/bcc
    if any(ci > 0 for ci in c_list):
        c_avg = float(np.dot(c_list, p_arr) / p_arr.sum())
    else:
        c_avg = None
    return a_avg, c_avg

def build_bulk_cell(structure: str, a_avg: float, c_avg: Optional[float]):
    if structure == "fcc":
        return FaceCenteredCubic("X", latticeconstant=a_avg)
    elif structure == "bcc":
        return BodyCenteredCubic("X", latticeconstant=a_avg)
    elif structure == "hcp":
        c_over_a = (c_avg / a_avg) if (c_avg and a_avg) else DEFAULT_HCP_COA
        return HexagonalClosedPacked("X", latticeconstant=a_avg, c_over_a=c_over_a)
    else:
        raise ValueError("structure must be fcc/bcc/hcp")

def generate_alloy_surface(elements, proportions, a_avg, c_avg,
                           structure, miller, layers, vacuum, supercell_dims):
    cell = build_bulk_cell(structure, a_avg, c_avg)
    slab = surface(cell, miller, layers, vacuum=vacuum)
    slab = slab.repeat((supercell_dims[0], supercell_dims[1], 1))

    total = len(slab)
    float_nums = proportions * total
    int_nums = np.floor(float_nums).astype(int)
    diff = total - int_nums.sum()
    fracs = float_nums - int_nums
    indices = np.argsort(-fracs)
    for i in range(diff):
        int_nums[indices[i]] += 1

    symbols = sum([[e] * cnt for e, cnt in zip(elements, int_nums)], [])
    random.shuffle(symbols)
    slab.set_chemical_symbols(symbols)

    uniq, counts = np.unique(symbols, return_counts=True)
    actual_comp = {u: c / total for u, c in zip(uniq, counts)}
    return slab, actual_comp

def calc_mixing_entropy(proportions: np.ndarray) -> float:
    return float(-R_GAS * np.sum(proportions * np.log(proportions + 1e-16)))

def calc_delta(proportions: np.ndarray, elements: list) -> float:
    radii = np.array([PERIODIC[e]["r"] for e in elements], float)
    r_avg = float(np.sum(proportions * radii))
    delta = 100.0 * math.sqrt(float(np.sum(proportions * (1.0 - radii / r_avg) ** 2)))
    return delta

def calc_mixing_enthalpy(elements: list, proportions: np.ndarray, H_df: Optional["pd.DataFrame"]) -> float:
    if H_df is None:
        return float("nan")
    H_mix = 0.0
    for i, ei in enumerate(elements):
        for j in range(i + 1, len(elements)):
            ej = elements[j]
            try:
                Hij = H_df.loc[ei, ej]
            except Exception:
                try:
                    Hij = H_df.loc[ej, ei]
                except Exception:
                    Hij = 0.0  # fallback
                    warnings.warn(f"[Enthalpy] Missing pair ({ei},{ej}) — using 0.0 kJ/mol.")
            H_mix += 4.0 * proportions[i] * proportions[j] * Hij
    return float(H_mix)

def calc_omega(Tm: float, S_mix: float, H_mix: float) -> float:
    if (H_mix is None) or (not np.isfinite(H_mix)) or (abs(H_mix) < 1e-12):
        return float("nan")
    return float((Tm * S_mix) / abs(H_mix * 1e3))  # H_mix kJ/mol -> J/mol

def weighted_melting_point(elements: list, proportions: np.ndarray) -> float:
    Tms = np.array([PERIODIC[e]["Tm"] for e in elements], float)
    return float(np.sum(proportions * Tms))

def monte_carlo_sampling(allowed: list, structure: str, H_df: Optional["pd.DataFrame"],
                         min_el: int, max_el: int,
                         smin: float, deltamax: float,
                         hmin: float, hmax: float,
                         omegamin: float,
                         max_attempts: int = 10000):
    for _ in range(max_attempts):
        n = random.randint(min_el, max_el)
        chosen = random.sample(allowed, n)
        comps = np.random.dirichlet(np.ones(n) * 2.0)
        S = calc_mixing_entropy(comps)
        delta = calc_delta(comps, chosen)
        H = calc_mixing_enthalpy(chosen, comps, H_df)
        Tm = weighted_melting_point(chosen, comps)
        Omega = calc_omega(Tm, S, H)

        # Screen (if enthalpy not provided, Omega is NaN — ignore that criterion)
        pass_omega = (not np.isfinite(Omega)) or (Omega > omegamin)
        if (S > smin) and (delta < deltamax) and (np.isnan(H) or (hmin < H < hmax)) and pass_omega:
            a_avg, c_avg = average_lattice_constants(chosen, comps, structure)
            return chosen, comps, S, delta, H, Omega, a_avg, c_avg
    raise RuntimeError("No valid composition found within max attempts.")

def write_summary_line(csv_file, filename, structure, elems, comps, actual, S, delta, H, Omega, a_avg, c_avg):
    elems_csv = ";".join(elems)
    target_csv = ";".join(f"{v:.4f}" for v in comps)
    actual_csv = ";".join(f"{actual.get(e, 0.0):.4f}" for e in elems)
    line = (
        f"{filename},{structure},{elems_csv},{target_csv},{actual_csv},"
        f"{S if S is not None else float('nan'):.4f},"
        f"{delta if delta is not None else float('nan'):.4f},"
        f"{H if H is not None and np.isfinite(H) else float('nan'):.4f},"
        f"{Omega if Omega is not None and np.isfinite(Omega) else float('nan'):.4f},"
        f"{a_avg:.4f},{(c_avg if c_avg is not None else 0.0):.4f}\n"
    )
    csv_file.write(line)

def main():
    ap = argparse.ArgumentParser(description="Flexible HEA structure generator")
    # geometry
    ap.add_argument("--structure", type=str, default="fcc", choices=["fcc", "bcc", "hcp"], help="Lattice type")
    ap.add_argument("--miller", type=int, nargs=3, default=[1, 1, 1], help="Miller indices")
    ap.add_argument("--layers", type=int, default=4, help="Number of atomic layers")
    ap.add_argument("--vacuum", type=float, default=10.0, help="Vacuum thickness (Å)")
    ap.add_argument("--supercell", type=int, nargs=2, default=[2, 3], help="In-plane supercell size (nx ny)")
    # output
    ap.add_argument("--output", type=str, default="./HEA_slabs", help="Output directory")
    ap.add_argument("--num_alloys", type=int, default=1, help="Number of alloys to generate")
    # fixed mode
    ap.add_argument("--fixed-elements", type=str, default=None,
                    help="Comma-separated fixed elements (e.g., Pt,Fe,Co,Ni,Cu)")
    ap.add_argument("--fixed-composition", type=str, default=None,
                    help="Comma-separated fractions (sum≈1), e.g., 0.40,0.1625,0.125,0.1625,0.15")
    # random mode
    ap.add_argument("--elements", type=str, default=None,
                    help="Comma-separated allowed element pool for random sampling; default=all PERIODIC keys")
    ap.add_argument("--min-elements", type=int, default=5, help="Min number of elements (random mode)")
    ap.add_argument("--max-elements", type=int, default=7, help="Max number of elements (random mode)")
    ap.add_argument("--max-attempts", type=int, default=10000, help="Max Monte Carlo attempts")
    # screening thresholds (random mode)
    ap.add_argument("--Smin_over_R", type=float, default=1.6, help="S/R threshold (default 1.6)")
    ap.add_argument("--delta_max", type=float, default=6.6, help="delta (%%) max")
    ap.add_argument("--Hmin", type=float, default=-15.0, help="H_mix min (kJ/mol)")
    ap.add_argument("--Hmax", type=float, default=5.0, help="H_mix max (kJ/mol)")
    ap.add_argument("--Omega_min", type=float, default=1.1, help="Omega min")
    # enthalpy matrix (optional)
    ap.add_argument("--enthalpy_file", type=str, default=None, help="Excel file (pairwise mixing enthalpy matrix)")
    ap.add_argument("--enthalpy_sheet", type=int, default=0, help="Sheet index for the enthalpy matrix")
    # overrides
    ap.add_argument("--override_a", type=float, default=None, help="Override final lattice a_avg (Å)")
    ap.add_argument("--override_c", type=float, default=None, help="Override final lattice c_avg (Å, for hcp)")

    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    struct = args.structure
    miller_idx = tuple(args.miller)
    sc_dims = tuple(args.supercell)

    # Load enthalpy matrix if provided
    H_df = None
    if args.enthalpy_file:
        if pd is None:
            raise RuntimeError("pandas is required when --enthalpy_file is given.")
        H_df = pd.read_excel(args.enthalpy_file, sheet_name=args.enthalpy_sheet, index_col=0)

    csv_path = os.path.join(args.output, "alloy_parameters.csv")
    with open(csv_path, "w") as csv_file:
        csv_file.write("Filename,Structure,Elements,Target_Composition,Actual_Composition,"
                       "S_mix,Delta,H_mix,Omega,a_avg,c_avg\n")

        # Fixed-composition mode
        if args.fixed_elements and args.fixed_composition:
            elems = parse_list_str(args.fixed_elements)
            comps = np.array([float(x) for x in parse_list_str(args.fixed_composition)], float)
            if len(elems) != len(comps):
                raise ValueError("Length of --fixed-elements and --fixed-composition must match.")
            comps = normalize_comp(comps)

            # Lattice averaging (with per-element estimation as needed)
            a_avg, c_avg = average_lattice_constants(elems, comps, struct)

            # Optional override of final a/c
            if args.override_a is not None:
                a_avg = float(args.override_a)
            if struct == "hcp" and args.override_c is not None:
                c_avg = float(args.override_c)

            # Build slab
            slab, actual = generate_alloy_surface(
                elems, comps, a_avg, c_avg, struct, miller_idx, args.layers, args.vacuum, sc_dims
            )

            # Compute descriptors if possible
            S = calc_mixing_entropy(comps)
            delta = calc_delta(comps, elems)
            H = calc_mixing_enthalpy(elems, comps, H_df)
            Tm = weighted_melting_point(elems, comps)
            Omega = calc_omega(Tm, S, H)

            comp_str = "_".join(f"{e}{p*100:.2f}" for e, p in zip(elems, comps))
            safe_str = re.sub(r"[^\w\-\. ]", "_", comp_str)
            filename = f"{struct}_{''.join(map(str, miller_idx))}_{safe_str}.vasp"
            fpath = os.path.join(args.output, filename)
            write_vasp(fpath, slab, direct=True, vasp5=True, sort=True)

            write_summary_line(csv_file, filename, struct, elems, comps, actual, S, delta, H, Omega, a_avg, c_avg)
            print(f"[1/1] Generated fixed alloy: {filename}")
            return

        # Random mode
        allowed = parse_list_str(args.elements) if args.elements else list(PERIODIC.keys())
        for i in range(args.num_alloys):
            try:
                S_over_R = args.Smin_over_R
                Smin = S_over_R * R_GAS

                (elems, comps, S, delta, H, Omega, a_avg, c_avg) = monte_carlo_sampling(
                    allowed=allowed,
                    structure=struct,
                    H_df=H_df,
                    min_el=args.min_elements,
                    max_el=args.max_elements,
                    smin=Smin,
                    deltamax=args.delta_max,
                    hmin=args.Hmin,
                    hmax=args.Hmax,
                    omegamin=args.Omega_min,
                    max_attempts=args.max_attempts,
                )

                # override if provided
                if args.override_a is not None:
                    a_avg = float(args.override_a)
                if struct == "hcp" and args.override_c is not None:
                    c_avg = float(args.override_c)

                slab, actual = generate_alloy_surface(
                    elems, comps, a_avg, c_avg, struct, miller_idx, args.layers, args.vacuum, sc_dims
                )

                comp_str = "_".join(f"{e}{p*100:.2f}" for e, p in zip(elems, comps))
                safe_str = re.sub(r"[^\w\-\. ]", "_", comp_str)
                filename = f"{struct}_{''.join(map(str, miller_idx))}_{safe_str}.vasp"
                fpath = os.path.join(args.output, filename)
                write_vasp(fpath, slab, direct=True, vasp5=True, sort=True)

                write_summary_line(csv_file, filename, struct, elems, comps, actual, S, delta, H, Omega, a_avg, c_avg)
                print(f"[{i+1}/{args.num_alloys}] Generated: {filename}")

            except Exception as e:
                warnings.warn(f"[ERROR] Alloy {i+1} failed: {e}")

if __name__ == "__main__":
    main()

