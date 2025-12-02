# -*- coding: utf-8 -*-
"""
compute_eoc_empband_dd_v2.py

Key Changes:
- Global DD: Outputs eps=0 / 0.05 / 0.10 (column names: eps0, eps005, eps010)
- Conditional DD: Computed only within "Overlap Windows" (EmpBand q=0.866; SB k=1.5; optional k=1)
- Still Outputs: EOC, EmpBand(q=0.68/0.866), ShuttleBand(k=1/1.5), NOF(±0.05/±0.10)

Dependencies: numpy, pandas
Example (Batch processing multiple pairs):
  python compute_eoc_empband_dd_v2.py ^
    --pair HEA1=HEA1_peak1.csv,HEA1_peak2.csv ^
    --out HEA1_metrics_v2.csv --energy-col "Prediction"

Switchable DD Method:
  --dd-method gaussian|kde|empirical   (default: gaussian; conditional DD defaults to kde internally)

Selectable Conditional Windows:
  --cond-windows empband866,sbk15,sbk1    (default: empband866,sbk15)

Custom DD Thresholds (comma-separated):
  --dd-eps 0,0.05,0.10
"""

import argparse
import math
import os
import numpy as np
import pandas as pd

# ---------- Basic Utilities ----------

def robust_read_csv(path, encoding='utf-8-sig'):
    tries = [encoding, 'utf-8', 'gbk', 'gb18030', 'utf-8-sig']
    for enc in tries:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"Failed to read CSV: {path}")

def pick_energy_series(df: pd.DataFrame, energy_col: str = None) -> np.ndarray:
    # Select ΔGH* column. Defaults to 'Secondary Prediction' or Chinese equivalents.
    candidates = []
    if energy_col:
        candidates.append(energy_col)
    # Keep Chinese column names for compatibility with existing datasets
    candidates += ['Prediction']
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors='coerce').astype(float).values
            s = s[np.isfinite(s)]
            return s
    raise KeyError(f"Energy column not found. Please specify using --energy-col. Available columns: {list(df.columns)}")

def mean_std(arr: np.ndarray):
    if len(arr) == 0:
        return (np.nan, np.nan)
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mu, sd

def silverman_bandwidth(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return 1.0
    std = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = max(q75 - q25, 1e-12)
    scale = min(std, iqr / 1.34) if std > 0 else iqr / 1.34
    h = 0.9 * scale * (n ** (-1/5))
    return max(h, 1e-3)

def gaussian_kde_on_grid(samples: np.ndarray, grid: np.ndarray, bandwidth: float = None):
    s = np.asarray(samples, dtype=float)
    g = np.asarray(grid, dtype=float)
    n = len(s)
    if n == 0:
        return np.zeros_like(g)
    if bandwidth is None:
        bandwidth = silverman_bandwidth(s)
    diff = (g[:, None] - s[None, :]) / bandwidth
    phi = np.exp(-0.5 * diff * diff) / math.sqrt(2 * math.pi)
    dens = np.mean(phi, axis=1) / bandwidth
    # Normalization
    area = np.trapz(dens, g)
    if area > 0:
        dens = dens / area
    return dens

def build_energy_grid(surf: np.ndarray, sub: np.ndarray, step=1e-3):
    allx = np.concatenate([surf, sub])
    mu = np.mean(allx)
    sig = np.std(allx, ddof=1) if len(allx) > 1 else 0.1
    lo = min(allx.min(), mu - 6 * sig)
    hi = max(allx.max(), mu + 6 * sig)
    nmax = 40000
    npts = int(max(2000, min(nmax, max(2, math.ceil((hi - lo) / step)))))
    return np.linspace(lo, hi, npts)

def eoc_from_kde(surf: np.ndarray, sub: np.ndarray):
    grid = build_energy_grid(surf, sub, step=1e-3)
    fs = gaussian_kde_on_grid(surf, grid)
    fb = gaussian_kde_on_grid(sub, grid)
    val = np.trapz(np.minimum(fs, fb), grid)
    return float(np.clip(val, 0.0, 1.0))

def empband_from_samples(surf: np.ndarray, sub: np.ndarray, q: float):
    qs_lo = (1 - q) / 2
    qs_hi = (1 + q) / 2
    bs_lo, bs_hi = np.quantile(surf, [qs_lo, qs_hi])
    bsub_lo, bsub_hi = np.quantile(sub, [qs_lo, qs_hi])
    lo = max(bs_lo, bsub_lo)
    hi = min(bs_hi, bsub_hi)
    if lo >= hi:
        return 0.0, 0.0, 0.0, 0, 0, float(lo), float(hi)
    width = float(hi - lo)
    in_s = (surf >= lo) & (surf <= hi)
    in_b = (sub >= lo) & (sub <= hi)
    ms = float(in_s.sum() / len(surf))
    mb = float(in_b.sum() / len(sub))
    return width, ms, mb, int(in_s.sum()), int(in_b.sum()), float(lo), float(hi)

def shuttleband_pm_k_sigma(surf: np.ndarray, sub: np.ndarray, k: float):
    mu_s, sd_s = mean_std(surf)
    mu_b, sd_b = mean_std(sub)
    s_lo, s_hi = mu_s - k * sd_s, mu_s + k * sd_s
    b_lo, b_hi = mu_b - k * sd_b, mu_b + k * sd_b
    lo, hi = max(s_lo, b_lo), min(s_hi, b_hi)
    if lo >= hi:
        return 0.0, 0.0, 0.0, float(lo), float(hi)
    width = float(hi - lo)
    in_s = (surf >= lo) & (surf <= hi)
    in_b = (sub >= lo) & (sub <= hi)
    ms = float(in_s.sum() / len(surf))
    mb = float(in_b.sum() / len(sub))
    return ms, mb, width, float(lo), float(hi)

def dd_probability(sampA: np.ndarray, sampB: np.ndarray, eps: float, method='gaussian'):
    A = np.asarray(sampA, dtype=float)
    B = np.asarray(sampB, dtype=float)

    if method == 'empirical':
        if len(A) == 0 or len(B) == 0:
            return np.nan
        # O(n^2) empirical estimation
        AA = A[:, None]
        BB = B[None, :]
        return float(((AA - BB) >= eps).mean())

    elif method == 'kde':
        # ∫ f_A(e) F_B(e - eps) de
        grid = build_energy_grid(A, B, step=1e-3)
        fA = gaussian_kde_on_grid(A, grid)
        fB = gaussian_kde_on_grid(B, grid)
        dE = np.diff(grid).mean()
        FB = np.cumsum(fB) * dE
        FB = np.clip(FB, 0.0, 1.0)
        FB_shift = np.interp(grid - eps, grid, FB, left=0.0, right=1.0)
        val = float(np.trapz(fA * FB_shift, grid))
        return float(np.clip(val, 0.0, 1.0))

    else:  # gaussian (closed form)
        muA, sdA = mean_std(A)
        muB, sdB = mean_std(B)
        var = sdA ** 2 + sdB ** 2
        if var <= 1e-16:
            return float(1.0 if (muA - muB) >= eps else 0.0)
        z = (muA - muB - eps) / math.sqrt(var)
        return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))

def near_optimal_fraction(arr: np.ndarray, w: float):
    return float((np.abs(arr) <= w).sum() / len(arr)) if len(arr) else np.nan

def filter_region(arr: np.ndarray, lo: float, hi: float):
    return arr[(arr >= lo) & (arr <= hi)]

def eps_label(eps: float) -> str:
    # Naming convention: 0 -> eps0; 0.05 -> eps005; 0.10 -> eps010; others in meV (3 digits)
    if abs(eps) < 1e-12:
        return "eps0"
    if abs(eps - 0.05) < 1e-9:
        return "eps005"
    if abs(eps - 0.10) < 1e-9:
        return "eps010"
    return f"eps{int(round(eps * 1000)):03d}"

# ---------- Calculation for a Single Pair ----------

def compute_metrics_for_pair(
    surf_path: str,
    sub_path: str,
    hea_name: str,
    energy_col: str = None,
    encoding: str = 'utf-8-sig',
    dd_method: str = 'gaussian',
    dd_eps_list = (0.0, 0.05, 0.10),
    cond_windows = ('empband866','sbk15'),
    min_cond_count: int = 10
) -> dict:
    df_s = robust_read_csv(surf_path, encoding=encoding)
    df_b = robust_read_csv(sub_path, encoding=encoding)
    Es = pick_energy_series(df_s, energy_col=energy_col)
    Eb = pick_energy_series(df_b, energy_col=energy_col)

    # Basic Statistics
    mu_s, sd_s = mean_std(Es)
    mu_b, sd_b = mean_std(Eb)
    n_s, n_b = len(Es), len(Eb)

    # EOC
    eoc = eoc_from_kde(Es, Eb)

    # Global DD
    dd_vals = {}
    for eps in dd_eps_list:
        lab = eps_label(eps)
        dd_vals[f"DD_S2Sub_{lab}"] = dd_probability(Es, Eb, eps=eps, method=dd_method)
        dd_vals[f"DD_Sub2S_{lab}"] = dd_probability(Eb, Es, eps=eps, method=dd_method)

    # NOF
    nof_s_005 = near_optimal_fraction(Es, 0.05)
    nof_b_005 = near_optimal_fraction(Eb, 0.05)
    nof_s_010 = near_optimal_fraction(Es, 0.10)
    nof_b_010 = near_optimal_fraction(Eb, 0.10)

    # ShuttleBand k=1 / 1.5
    sb_s_mass_k1, sb_b_mass_k1, sb_w_k1, sb_lo_k1, sb_hi_k1 = shuttleband_pm_k_sigma(Es, Eb, k=1.0)
    sb_s_mass_k15, sb_b_mass_k15, sb_w_k15, sb_lo_k15, sb_hi_k15 = shuttleband_pm_k_sigma(Es, Eb, k=1.5)

    # EmpBand q=0.68 / 0.866
    eb68_w, eb68_ms, eb68_mb, eb68_cs, eb68_cb, eb68_lo, eb68_hi   = empband_from_samples(Es, Eb, q=0.68)
    eb866_w, eb866_ms, eb866_mb, eb866_cs, eb866_cb, eb866_lo, eb866_hi = empband_from_samples(Es, Eb, q=0.866)

    # Conditional DD (Within exchangeable energy regions; Windows: EmpBand866 / SBk15 / SBk1)
    cond_dd = {}
    # Use KDE internally (fits better after truncation)
    def add_cond_dd(tag: str, A: np.ndarray, B: np.ndarray):
        if len(A) >= min_cond_count and len(B) >= min_cond_count:
            for eps in dd_eps_list:
                lab = eps_label(eps)
                cond_dd[f"DD_S2Sub_{tag}_{lab}"] = dd_probability(A, B, eps=eps, method='kde')
                cond_dd[f"DD_Sub2S_{tag}_{lab}"] = dd_probability(B, A, eps=eps, method='kde')

    if 'empband866' in cond_windows and eb866_w > 0 and eb866_cs >= min_cond_count and eb866_cb >= min_cond_count:
        Es_R = filter_region(Es, eb866_lo, eb866_hi)
        Eb_R = filter_region(Eb, eb866_lo, eb866_hi)
        add_cond_dd('EmpBand866', Es_R, Eb_R)

    if 'sbk15' in cond_windows and sb_w_k15 > 0:
        Es_R2 = filter_region(Es, sb_lo_k15, sb_hi_k15)
        Eb_R2 = filter_region(Eb, sb_lo_k15, sb_hi_k15)
        add_cond_dd('SBk15', Es_R2, Eb_R2)

    if 'sbk1' in cond_windows and sb_w_k1 > 0:
        Es_R3 = filter_region(Es, sb_lo_k1, sb_hi_k1)
        Eb_R3 = filter_region(Eb, sb_lo_k1, sb_hi_k1)
        add_cond_dd('SBk1', Es_R3, Eb_R3)

    row = dict(
        HEA=hea_name,
        mu_surf_eV=mu_s, sigma_surf_eV=sd_s, n_surf=n_s,
        mu_sub_eV=mu_b,  sigma_sub_eV=sd_b, n_sub=n_b,
        EOC=eoc,
        # Global DD
        **dd_vals,
        # NOF
        NOF_Surf_w005=nof_s_005, NOF_Sub_w005=nof_b_005,
        NOF_Surf_w010=nof_s_010, NOF_Sub_w010=nof_b_010,
        # ShuttleBand
        SB_Surf_mass_k1=sb_s_mass_k1, SB_Sub_mass_k1=sb_b_mass_k1,
        SB_width_k1_eV=sb_w_k1, SB_lo_k1_eV=sb_lo_k1, SB_hi_k1_eV=sb_hi_k1,
        SB_Surf_mass_k15=sb_s_mass_k15, SB_Sub_mass_k15=sb_b_mass_k15,
        SB_width_k15_eV=sb_w_k15, SB_lo_k15_eV=sb_lo_k15, SB_hi_k15_eV=sb_hi_k15,
        # EmpBand 0.68 / 0.866
        EmpBand68_width_eV=eb68_w,
        EmpBand68_surf_mass=eb68_ms, EmpBand68_sub_mass=eb68_mb,
        EmpBand68_surf_count=eb68_cs, EmpBand68_sub_count=eb68_cb,
        EmpBand68_lo_eV=eb68_lo, EmpBand68_hi_eV=eb68_hi,
        EmpBand866_width_eV=eb866_w,
        EmpBand866_surf_mass=eb866_ms, EmpBand866_sub_mass=eb866_mb,
        EmpBand866_surf_count=eb866_cs, EmpBand866_sub_count=eb866_cb,
        EmpBand866_lo_eV=eb866_lo, EmpBand866_hi_eV=eb866_hi,
        # Conditional DD
        **cond_dd,
    )
    return row

# ---------- CLI ----------

def parse_pairs(pairs_list):
    out = []
    for item in pairs_list:
        if '=' not in item or ',' not in item:
            raise ValueError(f"Format error in --pair: {item}")
        hea, rest = item.split('=', 1)
        surf_csv, sub_csv = rest.split(',', 1)
        out.append((hea.strip(), surf_csv.strip(), sub_csv.strip()))
    return out

def main():
    ap = argparse.ArgumentParser(description="Compute EOC / EmpBand / DD (global & conditional) metrics from peak1/peak2 CSVs")
    ap.add_argument("--surf", type=str, help="Path to Surface (peak1) CSV (Single-pair mode)")
    ap.add_argument("--sub", type=str, help="Path to Subsurface (peak2) CSV (Single-pair mode)")
    ap.add_argument("--hea-name", type=str, help="HEA Name (Single-pair mode)")
    ap.add_argument("--pair", type=str, action="append",
                    help="Multi-pair mode: HEA=surf.csv,sub.csv (can be repeated)")
    ap.add_argument("--out", type=str, required=True, help="Path to output summary CSV")
    ap.add_argument("--energy-col", type=str, default="二次预测", help="Column name for ΔGH*")
    ap.add_argument("--encoding", type=str, default="utf-8-sig", help="Input encoding (default: utf-8-sig)")
    ap.add_argument("--dd-method", type=str, default="gaussian",
                    choices=["gaussian", "kde", "empirical"], help="Global DD calculation method (default: gaussian)")
    ap.add_argument("--dd-eps", type=str, default="0,0.05,0.10",
                    help="DD thresholds, comma-separated, e.g., 0,0.05,0.10")
    ap.add_argument("--cond-windows", type=str, default="empband866,sbk15",
                    help="Conditional DD windows, comma-separated: empband866,sbk15,sbk1")
    ap.add_argument("--min-cond-count", type=int, default=10, help="Minimum sample count in conditional window")
    args = ap.parse_args()

    # Parse eps list
    dd_eps_list = []
    for t in args.dd_eps.split(','):
        t = t.strip()
        if t:
            dd_eps_list.append(float(t))

    cond_windows = tuple([w.strip().lower() for w in args.cond_windows.split(',') if w.strip()])

    rows = []
    if args.pair:
        pairs = parse_pairs(args.pair)
        for hea, surf_csv, sub_csv in pairs:
            rows.append(
                compute_metrics_for_pair(
                    surf_csv, sub_csv, hea_name=hea,
                    energy_col=args.energy_col, encoding=args.encoding,
                    dd_method=args.dd_method, dd_eps_list=dd_eps_list,
                    cond_windows=cond_windows, min_cond_count=args.min_cond_count
                )
            )
    else:
        if not (args.surf and args.sub and args.hea_name):
            ap.error("Single-pair mode requires --surf, --sub, and --hea-name; or use --pair for multi-pair mode.")
        rows.append(
            compute_metrics_for_pair(
                args.surf, args.sub, hea_name=args.hea_name,
                energy_col=args.energy_col, encoding=args.encoding,
                dd_method=args.dd_method, dd_eps_list=dd_eps_list,
                cond_windows=cond_windows, min_cond_count=args.min_cond_count
            )
        )

    df = pd.DataFrame(rows)

    # Expected column order (Missing columns will be filled with NaN)
    base_cols = [
        "HEA",
        "mu_surf_eV","sigma_surf_eV","n_surf",
        "mu_sub_eV","sigma_sub_eV","n_sub",
        "EOC",
        # Global DD:
        "DD_S2Sub_eps0","DD_Sub2S_eps0",
        "DD_S2Sub_eps005","DD_Sub2S_eps005",
        "DD_S2Sub_eps010","DD_Sub2S_eps010",
        # NOF:
        "NOF_Surf_w005","NOF_Sub_w005","NOF_Surf_w010","NOF_Sub_w010",
        # ShuttleBand:
        "SB_Surf_mass_k1","SB_Sub_mass_k1","SB_width_k1_eV","SB_lo_k1_eV","SB_hi_k1_eV",
        "SB_Surf_mass_k15","SB_Sub_mass_k15","SB_width_k15_eV","SB_lo_k15_eV","SB_hi_k15_eV",
        # EmpBand:
        "EmpBand68_width_eV","EmpBand68_surf_mass","EmpBand68_sub_mass",
        "EmpBand68_surf_count","EmpBand68_sub_count","EmpBand68_lo_eV","EmpBand68_hi_eV",
        "EmpBand866_width_eV","EmpBand866_surf_mass","EmpBand866_sub_mass",
        "EmpBand866_surf_count","EmpBand866_sub_count","EmpBand866_lo_eV","EmpBand866_hi_eV",
        # Conditional DD (EmpBand866 / SBk15 / SBk1):
        "DD_S2Sub_EmpBand866_eps0","DD_Sub2S_EmpBand866_eps0",
        "DD_S2Sub_EmpBand866_eps005","DD_Sub2S_EmpBand866_eps005",
        "DD_S2Sub_EmpBand866_eps010","DD_Sub2S_EmpBand866_eps010",
        "DD_S2Sub_SBk15_eps0","DD_Sub2S_SBk15_eps0",
        "DD_S2Sub_SBk15_eps005","DD_Sub2S_SBk15_eps005",
        "DD_S2Sub_SBk15_eps010","DD_Sub2S_SBk15_eps010",
        "DD_S2Sub_SBk1_eps0","DD_Sub2S_SBk1_eps0",
        "DD_S2Sub_SBk1_eps005","DD_Sub2S_SBk1_eps005",
        "DD_S2Sub_SBk1_eps010","DD_Sub2S_SBk1_eps010",
    ]
    df = df.reindex(columns=base_cols)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved -> {args.out}")

if __name__ == "__main__":
    main()

