#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Entropy Alloy (HEA) Random Structure Generator

This script generates random HEA surface structures based on specified criteria and writes VASP POSCAR files.
It uses Monte Carlo sampling with thermodynamic and geometric criteria (mixing entropy, atomic size mismatch,
mixing enthalpy, and Omega parameter) to select candidate compositions, then constructs slab models for each.
"""

import numpy as np
import random
import argparse
import pandas as pd
import os
import re
from ase.build import bulk, surface
from ase.io.vasp import write_vasp
from ase.lattice.hexagonal import HexagonalClosedPacked
from ase.lattice.cubic import FaceCenteredCubic, BodyCenteredCubic

# ---------------------------
# Element definitions & properties
# ---------------------------
element_properties = {
    'fcc': {
        'elements': ['Ni', 'Cu', 'Rh', 'Pd', 'Ag', 'Ir', 'Pt', 'Au'],
        'properties': {
            'Ni': {'lattice_constants': {'a': 3.52}, 'atomic_radius': 1.24, 'melting_point': 1728},
            'Cu': {'lattice_constants': {'a': 3.61}, 'atomic_radius': 1.28, 'melting_point': 1358},
            'Rh': {'lattice_constants': {'a': 3.80}, 'atomic_radius': 1.34, 'melting_point': 2237},
            'Pd': {'lattice_constants': {'a': 3.89}, 'atomic_radius': 1.37, 'melting_point': 1828},
            'Ag': {'lattice_constants': {'a': 4.09}, 'atomic_radius': 1.44, 'melting_point': 1235},
            'Ir': {'lattice_constants': {'a': 3.84}, 'atomic_radius': 1.36, 'melting_point': 2739},
            'Pt': {'lattice_constants': {'a': 3.92}, 'atomic_radius': 1.39, 'melting_point': 2041},
            'Au': {'lattice_constants': {'a': 4.08}, 'atomic_radius': 1.44, 'melting_point': 1337},
        },
    },
    'bcc': {
        'elements': ['V', 'Cr', 'Fe', 'Nb', 'Mo', 'Ta', 'W'],
        'properties': {
            'V':  {'lattice_constants': {'a': 3.03}, 'atomic_radius': 1.34, 'melting_point': 2183},
            'Cr': {'lattice_constants': {'a': 2.88}, 'atomic_radius': 1.28, 'melting_point': 2180},
            'Fe': {'lattice_constants': {'a': 2.87}, 'atomic_radius': 1.26, 'melting_point': 1811},
            'Nb': {'lattice_constants': {'a': 3.30}, 'atomic_radius': 1.46, 'melting_point': 2750},
            'Mo': {'lattice_constants': {'a': 3.15}, 'atomic_radius': 1.39, 'melting_point': 2896},
            'Ta': {'lattice_constants': {'a': 3.31}, 'atomic_radius': 1.46, 'melting_point': 3290},
            'W':  {'lattice_constants': {'a': 3.16}, 'atomic_radius': 1.39, 'melting_point': 3695},
        },
    },
    'hcp': {
        'elements': ['Sc', 'Ti', 'Co', 'Zn', 'Zr', 'Ru', 'Cd', 'Hf', 'Re', 'Os'],
        'properties': {
            'Sc': {'lattice_constants': {'a': 3.31, 'c': 5.27}, 'atomic_radius': 1.62, 'melting_point': 1814},
            'Ti': {'lattice_constants': {'a': 2.95, 'c': 4.68}, 'atomic_radius': 1.47, 'melting_point': 1941},
            'Co': {'lattice_constants': {'a': 2.51, 'c': 4.07}, 'atomic_radius': 1.25, 'melting_point': 1768},
            'Zn': {'lattice_constants': {'a': 2.66, 'c': 4.95}, 'atomic_radius': 1.33, 'melting_point': 693},
            'Zr': {'lattice_constants': {'a': 3.23, 'c': 5.15}, 'atomic_radius': 1.60, 'melting_point': 2128},
            'Ru': {'lattice_constants': {'a': 2.71, 'c': 4.28}, 'atomic_radius': 1.34, 'melting_point': 2607},
            'Cd': {'lattice_constants': {'a': 2.98, 'c': 5.62}, 'atomic_radius': 1.48, 'melting_point': 594},
            'Hf': {'lattice_constants': {'a': 3.19, 'c': 5.05}, 'atomic_radius': 1.59, 'melting_point': 2506},
            'Re': {'lattice_constants': {'a': 2.76, 'c': 4.46}, 'atomic_radius': 1.37, 'melting_point': 3459},
            'Os': {'lattice_constants': {'a': 2.74, 'c': 4.32}, 'atomic_radius': 1.35, 'melting_point': 3306},
        },
    },
}

# ---------------------------
# Physical property calculation functions
# ---------------------------
def calculate_mixing_entropy(proportions):
    """Calculate mixing entropy ΔS_mix (J/mol·K)"""
    R = 8.314
    return -R * np.sum(proportions * np.log(proportions))

def calculate_delta(proportions, atomic_radii):
    """Calculate atomic size mismatch δ (%)"""
    r_avg = np.sum(proportions * atomic_radii)
    return 100 * np.sqrt(np.sum(proportions * (1 - atomic_radii / r_avg)**2))

def calculate_mixing_enthalpy(elements, proportions, H_df):
    H_mix = 0.0
    missing_pairs = []
    for i, ei in enumerate(elements):
        for j in range(i+1, len(elements)):
            ej = elements[j]
            try:
                Hij = H_df.loc[ei, ej]
            except KeyError:
                try:
                    Hij = H_df.loc[ej, ei]
                except KeyError:
                    Hij = 0.0
                    missing_pairs.append((ei, ej))
            H_mix += 4 * proportions[i] * proportions[j] * Hij
    if missing_pairs:
        print("Warning: missing H_ij for pairs:", missing_pairs)
    return H_mix  # kJ/mol

def calculate_omega(T_m, S_mix, H_mix):
    if abs(H_mix) < 1e-9:
        return np.nan
    # kJ → J
    Hmix_J = H_mix * 1e3  
    return (T_m * S_mix) / abs(Hmix_J)

def average_lattice_constants(elements, proportions, structure):
    """Compute weighted average lattice constants a and c"""
    props = element_properties[structure]['properties']
    a_vals, c_vals = zip(*[
        (props[el]['lattice_constants'].get('a', 0) * p,
         props[el]['lattice_constants'].get('c', 0) * p)
        for el, p in zip(elements, proportions)
    ])
    a_avg = sum(a_vals) / sum(proportions)
    c_avg = sum(c_vals) / sum(proportions) if any(c_vals) else None
    return a_avg, c_avg

def random_proportions(n):
    """Generate random composition proportions that sum to 1"""
    return np.random.dirichlet(np.ones(n) * 2)

def monte_carlo_sampling(structure, enthalpy_df, min_el, max_el, max_attempts=10000):
    """
    Monte Carlo sampling for candidate HEA compositions.
    Returns elements list, composition, S_mix, δ, H_mix, Ω, a_avg, c_avg.
    """
    pool = element_properties[structure]['elements']
    props = element_properties[structure]['properties']
    for _ in range(max_attempts):
        n = random.randint(min_el, max_el)
        chosen = random.sample(pool, n)
        comps = random_proportions(n)
        S = calculate_mixing_entropy(comps)
        radii = np.array([props[e]['atomic_radius'] for e in chosen])
        δ = calculate_delta(comps, radii)
        H = calculate_mixing_enthalpy(chosen, comps, enthalpy_df)
        Tm = np.sum(comps * np.array([props[e]['melting_point'] for e in chosen]))
        Ω = calculate_omega(Tm, S, H)
        # Screening criteria
        if (S > 1.6 * 8.314) and (δ < 6.6) and (-15 < H < 5) and (Ω > 1.1):
            a_avg, c_avg = average_lattice_constants(chosen, comps, structure)
            return chosen, comps, S, δ, H, Ω, a_avg, c_avg
    raise RuntimeError("No valid composition found within max attempts")

def generate_alloy_surface(elements, proportions, a_avg, c_avg,
                            structure, miller, layers, vacuum, supercell_dims):
    """
    Build slab surface for the given HEA composition.
    Returns ASE atoms object and actual composition.
    """
    # Create bulk cell based on structure type
    if structure == 'fcc':
        cell = FaceCenteredCubic('X', latticeconstant=a_avg)
    elif structure == 'bcc':
        cell = BodyCenteredCubic('X', latticeconstant=a_avg)
    else:
        c_a_ratio = c_avg / a_avg if c_avg else 1.633
        cell = HexagonalClosedPacked('X', latticeconstant=a_avg, c_over_a=c_a_ratio)
    slab = surface(cell, miller, layers, vacuum=vacuum)
    slab = slab.repeat((*supercell_dims, 1))
    total = len(slab)
    float_nums = proportions * total
    int_nums = np.floor(float_nums).astype(int)
    diff = total - int_nums.sum()
    fracs = float_nums - int_nums
    indices = np.argsort(-fracs)
    for i in range(diff):
        int_nums[indices[i]] += 1
    symbols = sum([[el] * cnt for el, cnt in zip(elements, int_nums)], [])
    if len(symbols) < total:
        extra = indices[:total - len(symbols)]
        for idx in extra:
            symbols.append(elements[idx])
    random.shuffle(symbols)
    slab.set_chemical_symbols(symbols)
    uniq, counts = np.unique(symbols, return_counts=True)
    actual_comp = {u: c / total for u, c in zip(uniq, counts)}
    return slab, actual_comp

# ---------------------------
# Main: Argument parsing
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Random HEA structure generator with thermodynamic screening"
    )
    parser.add_argument("--num_alloys", type=int, default=1000,
                        help="Number of HEA structures to generate")
    parser.add_argument("--min_elements", type=int, default=5,
                        help="Minimum number of elements in each alloy")
    parser.add_argument("--max_elements", type=int, default=7,
                        help="Maximum number of elements in each alloy")
    parser.add_argument("--structures", type=str, default="fcc,bcc,hcp",
                        help="Comma-separated list of crystal structures to sample (e.g., fcc,bcc)")
    parser.add_argument("--miller", type=int, nargs=3, default=[1,1,1],
                        help="Miller indices for surface (three integers, e.g., 1 1 1)")
    parser.add_argument("--layers", type=int, default=4,
                        help="Number of atomic layers in slab")
    parser.add_argument("--vacuum", type=float, default=10.0,
                        help="Vacuum thickness in Ångströms")
    parser.add_argument("--supercell", type=int, nargs=2, default=[2,3],
                        help="Supercell dimensions in surface plane (e.g., 2 3)")
    parser.add_argument("--output", type=str, default="./HEA_slabs",
                        help="Output directory for POSCAR files and CSV summary")
    parser.add_argument("--enthalpy_file", type=str, required=True,
                        help="Path to Excel file with mixing enthalpy matrix (sheet indexed)")
    args = parser.parse_args()

    # Prepare output directory
    os.makedirs(args.output, exist_ok=True)
    enthalpy_df = pd.read_excel(args.enthalpy_file, sheet_name=0, index_col=0)
    struct_list = args.structures.split(',')
    miller_idx = tuple(args.miller)
    sc_dims = tuple(args.supercell)

    csv_path = os.path.join(args.output, "alloy_parameters.csv")
    with open(csv_path, 'w') as csv_file:
        csv_file.write("Filename,Structure,Elements,Target_Composition,Actual_Composition,"
                       "S_mix,Delta,H_mix,Omega,a_avg,c_avg\n")
        for i in range(args.num_alloys):
            try:
                struct = random.choice(struct_list)
                elems, comps, S, d_val, H, O, a_avg, c_avg = monte_carlo_sampling(
                    struct, enthalpy_df,
                    args.min_elements, args.max_elements
                )
                slab, actual = generate_alloy_surface(
                    elems, comps, a_avg, c_avg, struct,
                    miller_idx, args.layers, args.vacuum, sc_dims
                )
                # Construct filename
                comp_str = "_".join(f"{e}{p*100:.1f}" for e,p in zip(elems, comps))
                safe_str = re.sub(r"[^\w\-\. ]", "_", comp_str)
                filename = f"{struct}_{''.join(map(str,miller_idx))}_{safe_str}.vasp"
                file_path = os.path.join(args.output, filename)
                write_vasp(file_path, slab, direct=True, vasp5=True, sort=True)

                # Write summary line
                elems_csv = ";".join(elems)
                target_csv = ";".join(f"{v:.4f}" for v in comps)
                actual_csv = ";".join(f"{actual[e]:.4f}" for e in elems)
                line = (f"{filename},{struct},{elems_csv},{target_csv},{actual_csv},"
                        f"{S:.2f},{d_val:.2f},{H:.2f},{O:.2f},{a_avg:.3f},{c_avg or 0:.3f}\n")
                csv_file.write(line)
                print(f"[{i+1}/{args.num_alloys}] Generated: {filename}")
            except Exception as err:
                print(f"[ERROR] Alloy {i+1} failed: {err}")

if __name__ == '__main__':
    main()
