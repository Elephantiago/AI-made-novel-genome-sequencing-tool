#!/usr/bin/env python3
"""
DNA ALCHEMY FRAMEWORK v5.0 — COMPLETE REWRITE
==============================================

Changes from v4.1:
  [FIX 1]  Wavelet: switched to purine/pyrimidine binary signal + scale-normalised
           power.  Fallback no longer returns scale=2 for every genome.
  [FIX 2]  SVD Entanglement: replaced raw entropy (~6.74 for all) with spectral
           entropy ratio vs white-noise baseline — now genuinely discriminating.
  [FIX 3]  MFDFA: added minimum-length guard (≥1000 bp); PSTVd and other tiny
           sequences receive NaN instead of numerical garbage.
  [FIX 4]  Jaccard: default k raised to 7 (4^7=16384 possible k-mers vs 256 at
           k=4, eliminating near-universal saturation). k=4 still available via
           --jaccard-k 4 but raises a saturation warning.
  [FIX 5]  Persistent homology fallback: ε-neighbourhood now uses the 10th-
           percentile distance; returns meaningful Betti-0 and a loop-proxy.
  [FIX 6]  Fisher-Rao now emits per-organism distances to the centroid, not just
           a single mean across all pairs.
  [FIX 7]  __main__ block indentation bug corrected.
  [FIX 8]  Total dataset size now measured correctly; banner shows real MB.
  [NEW  1] Dinucleotide relative abundance (ρ) — classic, highly discriminating
           genomic signature.  Outputs the ρ-deviation score per genome.
  [NEW  2] k-mer Shannon entropy per organism.
  [NEW  3] GC-skew summary (mean, std, max deviation) from sliding windows.
  [NEW  4] Organism-type taxonomy dict with per-type colour coding in all plots.
  [NEW  5] Summary CSV + JSON saved to results dir at the end.
  [NEW  6] Composite radar-chart (spider plot) for all scalar metrics.
  [NEW  7] Correlation matrix of all numeric features (heatmap).
  [CLEAN]  Removed duplicate function definitions (plot_motif_logos vs
           build_community_logos), dead comments, and placeholder stubs.
  [CLEAN]  Proper logging with elapsed timestamps.
"""

import os, sys, time, zlib, argparse, warnings, json, csv, logging
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
def _adaptive_max_len(seq_len: int, default: int = 200_000) -> int:
    """Use up to 200k bp or 10% of the genome length, whichever is smaller."""
    return min(default, seq_len // 10)

# ══════════════════════════════════════════════════════════════
# 0.  CONSTANTS & TAXONOMY
# ══════════════════════════════════════════════════════════════

ACCESSIONS = {
    "phiX174":    "NC_001422",
    "pUC19":      "L09137",
    "SARS-CoV-2": "NC_045512",
    "M13":        "NC_003287",
    "pBR322":     "J01749",
    "SV40":       "NC_001669",
    "Lambda":     "NC_001416",
    "PSTVd":      "NC_002030",
    "Human_mtDNA":"NC_012920",
    "T7_phage":   "NC_001604",
    "MS2_phage":  "NC_001417",
}

LARGE_ACCESSIONS = {
    "E_coli_K12": "NC_000913",
    "B_subtilis":  "NC_000964",
}

EXTRA_ACCESSIONS = {
    "P_aeruginosa_PA O1":   "NC_002516",
    "M_tuberculosis_H37Rv": "NC_000962",
    "V_cholerae_chr1":      "NC_002505",
    "S_pneumoniae_R6":      "NC_003098",
    "H_influenzae_RdKW20":  "NC_000907",
    "M_jannaschii":         "NC_000916",
    "T_thermophilus":       "NC_006461",
    "S_cerevisiae_chrIV":   "NC_001136",
}

ULTRA_ACCESSIONS = {
    # Large bacteria & archaea (already good)
    "Salmonella_enterica":      "NC_003197",   # ~4.8 MB
    "Streptomyces_coelicolor":  "NC_005363",   # ~8.7 MB
    "Pseudomonas_putida":       "NC_009512",   # ~6.2 MB
    "Bacillus_cereus":          "NC_003909",   # ~5.4 MB
    "Rhodobacter_sphaeroides":  "NC_007779",   # ~4.6 MB
    "Sulfolobus_solfataricus":  "NC_002754",   # ~3.0 MB

    # Eukaryotic chromosomes
    "Arabidopsis_chr1":         "NC_003070",   # ~30 MB
    "Drosophila_chr2":          "NC_004353",   # ~25 MB
    "C_elegans_chrI":           "NC_003279",   # ~15 MB
    "S_cerevisiae_full":        "NC_001143",   # ~12 MB

    # === 4 New Human Chromosomes (well balanced) ===
    "Human_chr1":               "NC_000001",   # ~249 MB  (largest, gene-rich)
    "Human_chr2":               "NC_000002",   # ~243 MB  (second largest)
    "Human_chr19":              "NC_000019",   # ~59 MB   (high GC, gene-dense)
    "Human_chr21":              "NC_000021",   # ~48 MB   (smallest acrocentric, lower GC)

    # Keep one smaller human chromosome if you want even more balance
    # "Human_chr22":            "NC_000022",   # ~51 MB
}

ORGANISM_TYPES: Dict[str, str] = {
    "phiX174":              "phage",
    "pUC19":                "plasmid",
    "SARS-CoV-2":           "virus",
    "M13":                  "phage",
    "pBR322":               "plasmid",
    "SV40":                 "virus",
    "Lambda":               "phage",
    "PSTVd":                "viroid",
    "Human_mtDNA":          "organelle",
    "T7_phage":             "phage",
    "MS2_phage":            "phage",
    "E_coli_K12":           "bacterium",
    "B_subtilis":           "bacterium",
    "P_aeruginosa_PA O1":   "bacterium",
    "M_tuberculosis_H37Rv": "bacterium",
    "V_cholerae_chr1":      "bacterium",
    "S_pneumoniae_R6":      "bacterium",
    "H_influenzae_RdKW20":  "bacterium",
    "M_jannaschii":         "archaeon",
    "T_thermophilus":       "bacterium",
    "S_cerevisiae_chrIV":   "eukaryote",
}

TYPE_COLORS: Dict[str, str] = {
    "phage":     "#e41a1c",
    "virus":     "#ff7f00",
    "bacterium": "#377eb8",
    "archaeon":  "#984ea3",
    "eukaryote": "#4daf4a",
    "plasmid":   "#a65628",
    "viroid":    "#f781bf",
    "organelle": "#888888",
}

CACHE_DIR = "genome_cache"

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def _sep(title=""):
    w = 68
    if title:
        pad = (w - len(title) - 2) // 2
        log.info("─" * pad + f" {title} " + "─" * (w - pad - len(title) - 2))
    else:
        log.info("─" * w)

# ══════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════

def ensure_cache():
    os.makedirs(CACHE_DIR, exist_ok=True)


def fetch_sequence(name: str, accession: str, email: str) -> str:
    from Bio import Entrez, SeqIO
    cache_path = os.path.join(CACHE_DIR, f"{accession}.fasta")
    if os.path.exists(cache_path):
        log.info(f"  [cache]  {name:22s}  {accession}")
        record = SeqIO.read(cache_path, "fasta")
        return str(record.seq).upper().replace("U", "T")
    log.info(f"  [fetch]  {name:22s}  {accession} …")
    Entrez.email = email
    try:
        handle = Entrez.efetch(db="nucleotide", id=accession,
                               rettype="fasta", retmode="text")
        fasta_text = handle.read()
        handle.close()
    except Exception as exc:
        log.warning(f"  FAILED  {name}: {exc}")
        return ""
    with open(cache_path, "w") as f:
        f.write(fasta_text)
    from io import StringIO
    record = SeqIO.read(StringIO(fasta_text), "fasta")
    seq = str(record.seq).upper().replace("U", "T")
    log.info(f"           → {len(seq):,} bp")
    time.sleep(0.4)
    return seq

def load_all_sequences(email: str, full: bool = False,
                       extra: bool = False, ultra: bool = False) -> tuple:
    ensure_cache()
    acc_map = dict(ACCESSIONS)
    if full:
        acc_map.update(LARGE_ACCESSIONS)
    if extra:
        acc_map.update(LARGE_ACCESSIONS)
        acc_map.update(EXTRA_ACCESSIONS)
    if ultra:
        acc_map.update(LARGE_ACCESSIONS)
        acc_map.update(EXTRA_ACCESSIONS)
        acc_map.update(ULTRA_ACCESSIONS)

    names, seqs = [], []
    print("\n── Fetching sequences ──")
    for name, acc in acc_map.items():
        seq = fetch_sequence(name, acc, email)
        if seq:
            names.append(name)
            seqs.append(seq)
    total_mb = sum(len(s) for s in seqs) / 1_000_000
    print(f"   Loaded {len(seqs)} sequences  •  Total ~{total_mb:.1f} MB\n")
    return names, seqs

# ══════════════════════════════════════════════════════════════
# 2.  ENCODINGS (shared helpers)
# ══════════════════════════════════════════════════════════════

_BASE_COMPLEX = {"A": 1+0j, "C": 0+1j, "G": -1+0j, "T": 0-1j}
_BASE_INT     = {"A": 0.0,  "C": 1.0,  "G": 2.0,   "T": 3.0}


def _encode_complex(seq: str) -> np.ndarray:
    return np.array([_BASE_COMPLEX.get(b, 0) for b in seq], dtype=complex)


def _encode_int(seq: str) -> np.ndarray:
    return np.array([_BASE_INT.get(b, 1.5) for b in seq], dtype=float)


def _encode_purine(seq: str) -> np.ndarray:
    """Binary: +1=purine (A/G), -1=pyrimidine (C/T). Ideal for codon-period detection."""
    return np.array([1.0 if b in "AG" else -1.0 for b in seq], dtype=float)

# ══════════════════════════════════════════════════════════════
# 3.  ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════

# ── 3.1  FFT Spectral Analysis ───────────────────────────────

def fft_spectral_analysis(names: List[str], seqs: List[str]) -> Dict:
    """
    Maps bases to complex plane (A=+1, C=+i, G=−1, T=−i), computes FFT power
    spectrum.  Returns dominant frequency and peak amplitude ratio (signal/noise).
    The 1/3 Hz peak identifies strong codon-phase periodicity.
    """
    _sep("FFT Spectral Analysis")
    results = {}
    for name, seq in zip(names, seqs):
        adaptive_max = min(200_000, len(seq) // 10)
        s = seq[:adaptive_max]
        signal = _encode_complex(s)
        fft    = np.fft.fft(signal)
        half   = len(fft) // 2
        power  = np.abs(fft[1:half]) ** 2
        freqs  = np.fft.fftfreq(len(signal))[1:half]
        peak   = int(np.argmax(power))
        ratio  = float(power[peak] / (power.mean() + 1e-12))
        results[name] = {
            "dominant_freq":  round(float(abs(freqs[peak])), 5),
            "amplitude_ratio": round(ratio, 2),
            "codon_power_ratio": round(
                float(_codon_band_ratio(power, freqs)), 4),
        }
        log.info(f"   {name:22s}  freq={results[name]['dominant_freq']:.5f}"
                 f"  ratio={results[name]['amplitude_ratio']:8.2f}"
                 f"  codon_band={results[name]['codon_power_ratio']:.4f}")
    return results


def _codon_band_ratio(power: np.ndarray, freqs: np.ndarray) -> float:
    """Power in the ±0.02 band around 1/3 relative to total power."""
    mask = np.abs(np.abs(freqs) - 1/3) < 0.02
    if mask.sum() == 0:
        return 0.0
    return float(power[mask].sum() / (power.sum() + 1e-15))

# ── 3.2  Chaos Walk Fractal Dimension ────────────────────────

def _box_count(pts: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.array([len(np.unique((pts / s).astype(int), axis=0))
                     for s in scales], dtype=float)


def chaos_walk_fractal(names: List[str], seqs: List[str]) -> Dict:
    """
    2-D cumulative walk (purine/pyrimidine × keto/amino axes).
    Box-counting fractal dimension via least-squares log-log fit.
    Short sequences (<500 bp) receive NaN.
    """
    _sep("Chaos Walk Fractal Dimension")
    results = {}
    for name, seq in zip(names, seqs):
        max_len = _adaptive_max_len(len(seq))
        if len(seq) < 500:
            log.info(f"   {name:22s}  fractal_dim=NaN  (sequence too short)")
            results[name] = float("nan")
            continue
        walk_seq = seq[:max_len]
        x = np.cumsum([1 if b in "AG" else -1 for b in walk_seq])
        y = np.cumsum([1 if b in "AT" else -1 for b in walk_seq])
        pts = np.column_stack((x, y)).astype(float)

        max_exp = int(np.log2(len(pts))) - 2
        if max_exp < 3:
            results[name] = float("nan")
            continue

        scales = 2.0 ** np.arange(3, max_exp)
        counts = _box_count(pts, scales)
        valid = counts > 0
        dim = np.polyfit(np.log(1 / scales[valid]), np.log(counts[valid]), 1)[0]
        results[name] = round(float(dim), 4)
        log.info(f"   {name:22s}  fractal_dim={results[name]}  (used {max_len:,} bp)")
    return results


# ── 3.3  K-mer Jaccard Similarity ────────────────────────────

def kmer_jaccard(names: List[str], seqs: List[str], k: int = 7) -> np.ndarray:
    """
    Pairwise Jaccard similarity on k-mer SETS.
    Default k=7 (4^7=16384 possible k-mers) avoids the near-universal saturation
    that occurs at k=4 (4^4=256) for large genomes.  A saturation warning is
    printed if k<6.
    """
    _sep(f"K-mer Jaccard Similarity (k={k})")
    if k < 6:
        log.warning(f"   ⚠  k={k} has only {4**k} possible k-mers; large genomes "
                    f"will saturate the space → Jaccard inflated toward 1.0")
    ksets = [set(seq[i:i+k] for i in range(len(seq) - k + 1)
                 if all(b in "ACGT" for b in seq[i:i+k]))
             for seq in seqs]
    n   = len(names)
    mat = np.zeros((n, n))
    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            inter = len(ksets[i] & ksets[j])
            union = len(ksets[i] | ksets[j])
            mat[i, j] = mat[j, i] = inter / union if union else 0.0
    mean_off = mat[np.triu_indices(n, k=1)].mean()
    log.info(f"   Mean pairwise Jaccard (k={k}) = {mean_off:.4f}")
    return mat

# ── 3.4  LZ Complexity (zlib proxy) ──────────────────────────

def lz_complexity(names: List[str], seqs: List[str]) -> Dict:
    """
    zlib compression ratio as a length-sensitive complexity proxy.
    NOTE: Not length-normalised — short genomes (PSTVd) will appear less
    structured simply because they have less repeatable context for zlib.
    """
    _sep("LZ Complexity")
    results = {}
    for name, seq in zip(names, seqs):
        raw        = seq.encode()
        compressed = zlib.compress(raw, level=9)
        results[name] = round(len(compressed) / len(raw), 4)
        log.info(f"   {name:22s}  lz_ratio={results[name]}")
    return results

# ── 3.5  Wavelet Analysis (FIXED) ────────────────────────────

def _numpy_morlet_cwt(signal: np.ndarray, widths: np.ndarray,
                      w0: float = 6.0) -> np.ndarray:
    """Pure-numpy Morlet CWT.  Returns (n_scales, n_signal) complex array."""
    n   = len(signal)
    out = np.zeros((len(widths), n), dtype=complex)
    for i, s in enumerate(widths):
        half    = int(4 * s)
        t       = np.arange(-half, half + 1) / s
        wavelet = (np.exp(1j * w0 * t) * np.exp(-0.5 * t ** 2)
                   / (np.pi ** 0.25 * np.sqrt(s)))
        conv    = np.convolve(signal, wavelet[::-1].conj(), mode="full")
        start   = (len(conv) - n) // 2
        out[i]  = conv[start: start + n]
    return out


def wavelet_analysis(names: List[str], seqs: List[str]) -> Dict:
    """
    Morlet CWT on purine/pyrimidine binary signal with scale-normalised power.
    Adaptive max_len: uses more data from large genomes.
    """
    _sep("Wavelet Analysis (Morlet CWT)")
    try:
        import pywt
        use_pywt = True
        log.info("   [using PyWavelets]")
    except ImportError:
        use_pywt = False
        log.info("   [pywt not installed — using built-in numpy Morlet]")

    # Log-spaced widths 3..128 (covers codon and helix scales)
    widths = np.unique(
        np.round(np.logspace(np.log10(3), np.log10(128), 50)).astype(int)
    )

    results = {}
    for name, seq in zip(names, seqs):
        adaptive_max = _adaptive_max_len(len(seq))
        sig = _encode_purine(seq[:adaptive_max])
        sig = sig - sig.mean()

        if use_pywt:
            coef, _ = pywt.cwt(sig, widths, "cmor1.5-1.0")
        else:
            coef = _numpy_morlet_cwt(sig, widths)

        # Scale-normalised power (fixes the old constant scale=2 bug)
        power_raw = np.sum(np.abs(coef) ** 2, axis=1)
        power_norm = power_raw / (widths.astype(float) + 1e-10)

        dom_scale = int(widths[np.argmax(power_norm)])

        def _nearest_ratio(target):
            idx = int(np.argmin(np.abs(widths - target)))
            return float(power_norm[idx] / (power_norm.mean() + 1e-10))

        results[name] = {
            "dominant_scale_bp": dom_scale,
            "codon_scale_ratio": round(_nearest_ratio(3), 3),
            "helix_scale_ratio": round(_nearest_ratio(10), 3),
        }
        log.info(f"   {name:22s}  dom_scale={dom_scale:4d} bp"
                 f"  codon_ratio={results[name]['codon_scale_ratio']:.3f}"
                 f"  helix_ratio={results[name]['helix_scale_ratio']:.3f}")
    return results

# ── 3.6  Persistent Homology ──────────────────────────────────

def persistent_homology(names: List[str], seqs: List[str],
                        n_points: int = 400) -> Dict:
    """
    Vietoris-Rips PH on subsampled chaos-walk point clouds.
    Requires `ripser`; falls back to ε-neighbourhood graph (Betti-0 + loop proxy).
    The fallback now uses the 10th-percentile distance for ε, not the 5th,
    giving more stable connected-component counts.
    """
    _sep("Persistent Homology")
    try:
        from ripser import ripser as _ripser
        use_ripser = True
        log.info("   [using ripser]")
    except ImportError:
        use_ripser = False
        log.info("   [ripser not installed — using ε-neighbourhood fallback]")

    results = {}
    for name, seq in zip(names, seqs):
        x   = np.cumsum([1 if b in "AG" else -1 for b in seq])
        y   = np.cumsum([1 if b in "AT" else -1 for b in seq])
        pts = np.column_stack((x, y)).astype(float)
        idx = np.round(np.linspace(0, len(pts) - 1, n_points)).astype(int)
        pts = pts[idx]
        pts = (pts - pts.mean(0)) / (pts.std() + 1e-12)

        if use_ripser:
            dgms   = _ripser(pts, maxdim=1)["dgms"]
            betti0 = int(np.sum(np.isinf(dgms[0][:, 1])))
            betti1 = len(dgms[1])
            method = "ripser"
        else:
            dists = squareform(pdist(pts))
            eps   = float(np.percentile(dists[dists > 0], 10))
            G_ph  = nx.Graph()
            G_ph.add_nodes_from(range(len(pts)))
            rows, cols = np.where((dists < eps) & (dists > 0))
            G_ph.add_edges_from(zip(rows.tolist(), cols.tolist()))
            betti0 = nx.number_connected_components(G_ph)
            # Loop proxy: E - V + C  (Euler characteristic for each component)
            betti1 = max(0, G_ph.number_of_edges() - G_ph.number_of_nodes() + betti0)
            method = "fallback"

        results[name] = {"betti_0": betti0, "betti_1": betti1, "method": method}
        log.info(f"   {name:22s}  β₀={betti0}  β₁={betti1}  [{method}]")
    return results

# ── 3.7  Multifractal DFA (MFDFA) ────────────────────────────

def _mfdfa_single(seq: str, max_len: int = 200_000) -> Dict:
    """
    Multifractal DFA (Kantelhardt et al. 2002).
    Returns alpha_mean (Hölder exponent) and spectrum_width (Δα).
    Raises ValueError if sequence is too short for reliable results.
    """
    MIN_LEN = 1000
    s = seq[:max_len]
    if len(s) < MIN_LEN:
        raise ValueError(f"Sequence too short ({len(s)} bp < {MIN_LEN} required)")

    scales   = np.unique(np.logspace(1, np.log10(len(s) // 4), 20).astype(int))
    scales   = scales[scales > 4]
    q_values = np.linspace(-5, 5, 21)
    q_values = q_values[np.abs(q_values) > 0.1]

    profile = np.cumsum(np.array([1.0 if b in "AG" else -1.0 for b in s]) - 0.5)
    Fq      = np.full((len(q_values), len(scales)), np.nan)

    for si, sc in enumerate(scales):
        n_seg = len(profile) // sc
        if n_seg < 2:
            continue
        segs = profile[: n_seg * sc].reshape(n_seg, sc)
        x    = np.arange(sc)
        var  = np.array([np.mean((seg - np.polyval(np.polyfit(x, seg, 1), x)) ** 2)
                         for seg in segs])
        var  = np.maximum(var, 1e-15)
        for qi, q in enumerate(q_values):
            Fq[qi, si] = (np.mean(var ** (q / 2)) ** (1 / q) if q != 0
                          else np.exp(0.5 * np.mean(np.log(var))))

    hq = []
    valid_si = ~np.all(np.isnan(Fq), axis=0)
    log_s    = np.log(scales[valid_si])
    for qi in range(len(q_values)):
        row  = Fq[qi, valid_si]
        mask = ~np.isnan(row) & (row > 0)
        hq.append(np.polyfit(log_s[mask], np.log(row[mask]), 1)[0]
                  if mask.sum() > 2 else np.nan)
    hq = np.array(hq)
    return {
        "alpha_mean":    round(float(np.nanmean(hq)), 4),
        "spectrum_width": round(float(np.nanmax(hq) - np.nanmin(hq))
                                if np.sum(~np.isnan(hq)) > 1 else np.nan, 4),
    }


def multifractal_analysis(names: List[str], seqs: List[str]) -> Dict:
    _sep("Multifractal DFA")
    results = {}
    for name, seq in zip(names, seqs):
        adaptive_max = min(200_000, len(seq) // 10)
        try:
            res = _mfdfa_single(seq, max_len=adaptive_max)
        except ValueError as e:
            log.info(f"   {name:22s}  SKIPPED — {e}")
            res = {"alpha_mean": float("nan"), "spectrum_width": float("nan")}
        results[name] = res
        log.info(f"   {name:22s}  α_mean={res['alpha_mean']}  "
                 f"Δα={res['spectrum_width']}")
    return results

# ── 3.8  SVD Spectral Entropy (FIXED) ────────────────────────

def svd_entanglement(names: List[str], seqs: List[str],
                     window: int = 512, step: int = 256) -> Dict:
    """
    SVD spectral entropy ratio (fixed & adaptive).
    Uses more data from large genomes.
    """
    _sep("SVD Spectral Entropy (scale-normalised)")
    results = {}
    for name, seq in zip(names, seqs):
        adaptive_max = _adaptive_max_len(len(seq))
        signal = _encode_int(seq[:adaptive_max]) - 2.0   # centre around 0
        m = window // 2
        ratios = []
        for start in range(0, len(signal) - window, step):
            chunk = signal[start:start + window]
            if len(chunk) < window:
                continue
            hankel = np.array([chunk[i:i + (window - m + 1)]
                               for i in range(m)])
            sv = np.linalg.svd(hankel, compute_uv=False)
            sv = sv[sv > 1e-12]
            if len(sv) == 0:
                continue
            p = sv / sv.sum()
            obs_H = float(-np.dot(p, np.log2(p + 1e-15)))
            ref_H = np.log2(min(m, window - m + 1))
            ratios.append(obs_H / (ref_H + 1e-10))
        results[name] = round(float(np.mean(ratios)), 5) if ratios else float("nan")
        log.info(f"   {name:22s}  svd_entropy_ratio={results[name]}")
    return results

# ── 3.9  Fisher-Rao Geometry ──────────────────────────────────

def fisher_rao_geometry(names: List[str], seqs: List[str], k: int = 3
                        ) -> Tuple[np.ndarray, Dict]:
    """
    Pairwise Fisher-Rao geodesic distances on normalised k-mer frequency vectors.
    Also returns per-organism distance to the centroid frequency vector,
    which is a concise measure of how 'unusual' a genome's composition is.
    """
    _sep(f"Fisher-Rao Geometry (k={k})")
    all_kmers = sorted({seq[i:i+k] for seq in seqs for i in range(len(seq)-k+1)
                        if all(b in "ACGT" for b in seq[i:i+k])})
    kmer_idx  = {km: i for i, km in enumerate(all_kmers)}
    n, m      = len(seqs), len(all_kmers)
    freq_mat  = np.zeros((n, m))
    for i, seq in enumerate(seqs):
        for j in range(len(seq) - k + 1):
            km = seq[j: j+k]
            if km in kmer_idx:
                freq_mat[i, kmer_idx[km]] += 1
        freq_mat[i] /= freq_mat[i].sum() + 1e-15

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            inner      = np.clip(np.dot(np.sqrt(freq_mat[i]),
                                        np.sqrt(freq_mat[j])), -1, 1)
            dist[i, j] = dist[j, i] = 2 * np.arccos(inner)

    centroid  = freq_mat.mean(0)
    centroid /= centroid.sum() + 1e-15
    centroid_dist = {}
    for i, name in enumerate(names):
        inner = np.clip(np.dot(np.sqrt(freq_mat[i]), np.sqrt(centroid)), -1, 1)
        centroid_dist[name] = round(float(2 * np.arccos(inner)), 4)

    mean_d = dist[np.triu_indices(n, k=1)].mean()
    log.info(f"   Mean pairwise distance = {mean_d:.4f}")
    for name in names:
        log.info(f"   {name:22s}  dist_to_centroid={centroid_dist[name]}")
    return dist, centroid_dist

# ── 3.10  Sheaf Cohomology ────────────────────────────────────

def sheaf_cohomology_approx(names: List[str], seqs: List[str],
                            base_patch_size: int = 200) -> Dict:
    """
    Čech-style H¹ obstruction with adaptive patch size.
    """
    _sep("Sheaf Cohomology")
    results = {}
    for name, seq in zip(names, seqs):
        # Adaptive patch size: bigger patches for longer genomes
        patch_size = min(base_patch_size, len(seq) // 50)
        overlap = patch_size // 2
        step = patch_size - overlap
        patches = [seq[i:i + patch_size]
                   for i in range(0, len(seq) - patch_size + 1, step)]
        if len(patches) < 3:
            results[name] = float("nan")
            continue

        def freq_vec(patch):
            c = Counter(patch[j:j+3] for j in range(len(patch)-3+1)
                        if all(b in "ACGT" for b in patch[j:j+3]))
            v = np.array(list(c.values()), dtype=float)
            return v / (v.sum() + 1e-15)

        inconsistencies = []
        for i in range(len(patches) - 2):
            fa, fb, fc = freq_vec(patches[i]), freq_vec(patches[i+1]), freq_vec(patches[i+2])
            L = min(len(fa), len(fb), len(fc))
            if L == 0:
                continue
            fa, fb, fc = fa[:L], fb[:L], fc[:L]
            cos = lambda a, b: np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-15)
            inconsistencies.append(abs(cos(fa, fb) * cos(fb, fc) - cos(fa, fc)))

        results[name] = round(float(np.mean(inconsistencies)), 5) if inconsistencies else float("nan")
        log.info(f"   {name:22s}  H1_obstruction={results[name]}")
    return results

# ── 3.11  GF(4) Polynomial Entropy ───────────────────────────

def gf4_degree_distribution(names: List[str], seqs: List[str],
                             window: int = 60) -> Dict:
    """
    GF(4) polynomial degree-distribution entropy.  Each window is treated as a
    coefficient list; entropy of the degree distribution measures polynomial
    complexity.
    """
    _sep("GF(4) Degree Distribution Entropy")
    GF4 = {"A": 0, "C": 1, "G": 2, "T": 3}
    results = {}
    for name, seq in zip(names, seqs):
        degrees = []
        for i in range(0, len(seq) - window + 1, window):
            coeffs = [GF4.get(b, 0) for b in seq[i: i+window]]
            nz     = [j for j, c in enumerate(coeffs) if c != 0]
            if nz:
                degrees.append(nz[-1])
        if degrees:
            cnt   = Counter(degrees)
            total = sum(cnt.values())
            probs = np.array([v / total for v in cnt.values()])
            h     = float(-np.dot(probs, np.log2(probs + 1e-15)))
        else:
            h = float("nan")
        results[name] = round(h, 4)
        log.info(f"   {name:22s}  gf4_entropy={results[name]}")
    return results

# ── 3.12  [NEW] Dinucleotide Relative Abundance ───────────────

def dinucleotide_relative_abundance(names: List[str], seqs: List[str]) -> Dict:
    """
    Computes ρ(XY) = f(XY) / (f(X) · f(Y)) for all 16 dinucleotides.
    Returns the mean absolute deviation of ρ from 1.0 as a scalar per genome
    (high deviation = strong dinucleotide bias, e.g. CpG suppression in mammals).
    Also returns the most over- and under-represented dinucleotides.
    """
    _sep("Dinucleotide Relative Abundance")
    BASES = list("ACGT")
    results = {}
    for name, seq in zip(names, seqs):
        mono = Counter(seq)
        di   = Counter(seq[i:i+2] for i in range(len(seq)-1)
                       if all(b in "ACGT" for b in seq[i:i+2]))
        total_mono = sum(mono[b] for b in BASES)
        total_di   = sum(di.values())
        rho = {}
        for x in BASES:
            for y in BASES:
                fxy = di.get(x+y, 0) / (total_di + 1e-15)
                fx  = mono.get(x, 0) / (total_mono + 1e-15)
                fy  = mono.get(y, 0) / (total_mono + 1e-15)
                rho[x+y] = fxy / (fx * fy + 1e-15)

        deviations  = {d: abs(v - 1.0) for d, v in rho.items()}
        mean_dev    = round(float(np.mean(list(deviations.values()))), 4)
        top_over    = max(rho, key=rho.get)
        top_under   = min(rho, key=rho.get)
        results[name] = {
            "rho_mean_deviation": mean_dev,
            "most_over":  f"{top_over}(ρ={rho[top_over]:.2f})",
            "most_under": f"{top_under}(ρ={rho[top_under]:.2f})",
            "rho": {d: round(v, 4) for d, v in rho.items()},
        }
        log.info(f"   {name:22s}  ρ_dev={mean_dev:.4f}"
                 f"  over={results[name]['most_over']}"
                 f"  under={results[name]['most_under']}")
    return results

# ── 3.13  [NEW] K-mer Shannon Entropy ────────────────────────

def kmer_entropy(names: List[str], seqs: List[str], k: int = 4) -> Dict:
    """
    Shannon entropy of the k-mer frequency distribution (bits).
    Maximum possible = log2(4^k) = 2k bits (all k-mers equally frequent).
    Normalised entropy = H / H_max in [0,1].
    """
    _sep(f"K-mer Shannon Entropy (k={k})")
    H_max   = 2 * k  # bits
    results = {}
    for name, seq in zip(names, seqs):
        cnt   = Counter(seq[i:i+k] for i in range(len(seq)-k+1)
                        if all(b in "ACGT" for b in seq[i:i+k]))
        total = sum(cnt.values())
        probs = np.array([v / total for v in cnt.values()])
        H     = float(-np.dot(probs, np.log2(probs + 1e-15)))
        results[name] = {
            "entropy_bits":       round(H, 4),
            "normalised_entropy": round(H / H_max, 4),
        }
        log.info(f"   {name:22s}  H={results[name]['entropy_bits']:.4f} bits"
                 f"  H_norm={results[name]['normalised_entropy']:.4f}")
    return results

# ── 3.14  [NEW] GC-Skew Analysis ─────────────────────────────

def gc_skew_analysis(names: List[str], seqs: List[str],
                     window: int = 10_000) -> Dict:
    """
    Sliding-window GC-skew = (G−C)/(G+C) and AT-skew = (A−T)/(A+T).
    Reports mean, std and max absolute deviation for each genome.
    Strong GC-skew indicates strand asymmetry (e.g. leading vs lagging replication).
    """
    _sep("GC-Skew Analysis")
    results = {}
    for name, seq in zip(names, seqs):
        gc_skews, at_skews = [], []
        for i in range(0, len(seq) - window + 1, window // 2):
            w = seq[i: i + window]
            G, C, A, T = w.count("G"), w.count("C"), w.count("A"), w.count("T")
            if G + C > 0:
                gc_skews.append((G - C) / (G + C))
            if A + T > 0:
                at_skews.append((A - T) / (A + T))
        if gc_skews:
            results[name] = {
                "gc_skew_mean":    round(float(np.mean(gc_skews)), 4),
                "gc_skew_std":     round(float(np.std(gc_skews)),  4),
                "gc_skew_max_abs": round(float(np.max(np.abs(gc_skews))), 4),
                "at_skew_mean":    round(float(np.mean(at_skews)), 4),
            }
        else:
            results[name] = {"gc_skew_mean": float("nan"),
                             "gc_skew_std":  float("nan"),
                             "gc_skew_max_abs": float("nan"),
                             "at_skew_mean": float("nan")}
        log.info(f"   {name:22s}  gc_skew_mean={results[name]['gc_skew_mean']:.4f}"
                 f"  gc_skew_std={results[name]['gc_skew_std']:.4f}")
    return results

# ── 3.15  GC Content + Correlations ──────────────────────────

def gc_content_and_correlations(names: List[str], seqs: List[str],
                                 invariants_dict: Dict) -> Dict:
    """GC% per genome + Pearson correlation with all provided scalar features."""
    _sep("GC Content & Correlations")
    gc = {}
    for name, seq in zip(names, seqs):
        gc[name] = round(100 * (seq.count("G") + seq.count("C")) / max(len(seq), 1), 2)
    log.info("   GC% per genome:")
    for n, v in gc.items():
        log.info(f"      {n:22s}  GC={v:5.2f}%")
    log.info("\n   Pearson correlations with GC%:")
    for feat, vals in invariants_dict.items():
        if not isinstance(vals, dict):
            continue
        fv  = np.array([float(vals.get(n, float("nan"))) for n in names])
        gcv = np.array([gc[n] for n in names])
        ok  = ~np.isnan(fv)
        if ok.sum() < 4:
            continue
        r, p = pearsonr(fv[ok], gcv[ok])
        sig  = " ← significant" if p < 0.05 else ""
        log.info(f"      {feat:20s}  r={r:+.3f}  p={p:.4f}{sig}")
    return gc

# ══════════════════════════════════════════════════════════════
# 4.  DE BRUIJN GRAPH
# ══════════════════════════════════════════════════════════════

def build_de_bruijn_weighted(names: List[str], seqs: List[str],
                             k: int = 4) -> nx.DiGraph:
    """Build aggregate weighted de Bruijn graph across all sequences."""
    _sep(f"Building Weighted de Bruijn Graph (k={k})")
    edge_counter: Counter = Counter()
    for seq in seqs:
        for i in range(len(seq) - k + 1):
            kmer = seq[i: i+k]
            if all(b in "ACGT" for b in kmer):
                edge_counter[kmer] += 1
    G = nx.DiGraph()
    for kmer, weight in edge_counter.items():
        G.add_edge(kmer[:-1], kmer[1:], weight=weight)
    log.info(f"   Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
    return G


def de_bruijn_analysis(G: nx.DiGraph, save_dir: str = ".") -> Dict:
    """
    Full de Bruijn analysis: topology, centralities, communities, sequence logos.
    """
    _sep("de Bruijn Deep Analysis")
    if G.number_of_nodes() == 0:
        return {}

    G_und = G.to_undirected()
    stats = {
        "nodes":    G.number_of_nodes(),
        "edges":    G.number_of_edges(),
        "density":  round(nx.density(G), 4),
        "num_scc":  len(list(nx.strongly_connected_components(G))),
    }
    degrees = [d for _, d in G.degree()]
    stats["avg_degree"] = round(float(np.mean(degrees)), 3)
    stats["max_degree"] = int(max(degrees))

    # Centralities
    pagerank   = nx.pagerank(G)
    between    = nx.betweenness_centrality(G)
    try:
        eigen  = nx.eigenvector_centrality(G, max_iter=1000)
    except Exception:
        eigen  = pagerank

    top_pr     = sorted(pagerank.items(),   key=lambda x: x[1], reverse=True)[:10]
    top_btwn   = sorted(between.items(),    key=lambda x: x[1], reverse=True)[:10]

    log.info(f"   Nodes={stats['nodes']}  Edges={stats['edges']}"
             f"  Density={stats['density']}  SCCs={stats['num_scc']}")
    log.info("   Top PageRank 3-mers:")
    for mer, pr in top_pr[:5]:
        log.info(f"      {mer}  PR={pr:.4f}")

    # Communities
    communities = list(nx.community.greedy_modularity_communities(G_und))
    log.info(f"   Found {len(communities)} communities")
    stats["num_communities"] = len(communities)

    _print_community_logos(G, communities)
    _plot_community_logos(G, communities, save_dir)
    _plot_community_graph(G, communities, save_dir)
    _plot_degree_distribution(G, save_dir)

    # Null-model Z-scores (10 replicates)
    n_rand, rand_clust = 10, []
    for _ in range(n_rand):
        rg = nx.erdos_renyi_graph(G.number_of_nodes(), nx.density(G), directed=False)
        rand_clust.append(nx.average_clustering(rg))
    obs_clust = nx.average_clustering(G_und)
    stats["z_clustering"] = round(
        (obs_clust - np.mean(rand_clust)) / (np.std(rand_clust) + 1e-8), 2)
    log.info(f"   Clustering Z-score vs Erdős-Rényi: {stats['z_clustering']}")

    return stats


def _print_community_logos(G: nx.DiGraph, communities) -> None:
    BASES = list("ACGT")
    for i, comm in enumerate(communities[:6]):
        if len(comm) < 3:
            continue
        subG    = G.subgraph(comm)
        top_mers = [mer for mer, _ in
                    sorted(subG.degree(), key=lambda x: x[1], reverse=True)[:12]]
        freq = np.zeros((3, 4))
        for mer in top_mers:
            for pos in range(min(3, len(mer))):
                if mer[pos] in BASES:
                    freq[pos, BASES.index(mer[pos])] += 1
        freq /= freq.sum(axis=1, keepdims=True) + 1e-8
        row_strs = []
        for pos in range(3):
            row = sorted(zip(BASES, freq[pos]), key=lambda x: x[1], reverse=True)
            row_strs.append("  ".join(f"{b}({p:.2f})" for b, p in row))
        log.info(f"   Community {i+1} ({len(comm)} nodes) logo:")
        for r in row_strs:
            log.info(f"      {r}")


def _plot_community_logos(G: nx.DiGraph, communities, save_dir: str) -> None:
    BASES = list("ACGT")
    COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, comm in enumerate(communities[:6]):
        if len(comm) < 3:
            continue
        subG     = G.subgraph(comm)
        top_mers = [mer for mer, _ in
                    sorted(subG.degree(), key=lambda x: x[1], reverse=True)[:12]]
        freq = np.zeros((3, 4))
        for mer in top_mers:
            for pos in range(min(3, len(mer))):
                if mer[pos] in BASES:
                    freq[pos, BASES.index(mer[pos])] += 1
        freq /= freq.sum(axis=1, keepdims=True) + 1e-8
        fig, ax = plt.subplots(figsize=(8, 3))
        x, w = np.arange(3), 0.2
        for j, b in enumerate(BASES):
            ax.bar(x + j*w - 0.3, freq[:, j], w, label=b,
                   color=COLORS[j], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(["Position 1", "Position 2", "Position 3"])
        ax.set_ylabel("Base frequency")
        ax.set_title(f"Community {i+1} Motif Logo")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"de_bruijn_community_{i+1}_logo.png"),
                    dpi=150)
        plt.close()


def _plot_community_graph(G: nx.DiGraph, communities, save_dir: str) -> None:
    COLORS = ["#e41a1c","#377eb8","#4daf4a","#ff7f00",
              "#984ea3","#a65628","#f781bf","#888888"]
    pos = nx.spring_layout(G, seed=42, k=0.5)
    fig = go.Figure()
    for i, comm in enumerate(communities):
        nx_list = list(comm)
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nx_list],
            y=[pos[n][1] for n in nx_list],
            mode="markers+text",
            marker=dict(size=9, color=COLORS[i % len(COLORS)]),
            text=nx_list, textposition="top center",
            name=f"Community {i+1}"))
    fig.update_layout(title="de Bruijn Graph — Communities Highlighted",
                      showlegend=True, height=700)
    fig.write_html(os.path.join(save_dir, "de_bruijn_communities.html"))
    log.info("   Saved de_bruijn_communities.html")


def _plot_degree_distribution(G: nx.DiGraph, save_dir: str) -> None:
    in_deg  = [d for _, d in G.in_degree()]
    out_deg = [d for _, d in G.out_degree()]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, vals, color, label in [
        (axs[0], in_deg,  "royalblue", "In-degree"),
        (axs[1], out_deg, "tomato",    "Out-degree"),
    ]:
        ax.hist(vals, bins=range(max(vals)+2), alpha=0.75,
                color=color, edgecolor="black")
        ax.set_title(f"{label} Distribution")
        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "de_bruijn_degree_distribution.png"), dpi=150)
    plt.close()
    log.info("   Saved de_bruijn_degree_distribution.png")

# ══════════════════════════════════════════════════════════════
# 5.  VISUALISATION LAYER
# ══════════════════════════════════════════════════════════════

def _org_color(name: str) -> str:
    return TYPE_COLORS.get(ORGANISM_TYPES.get(name, "bacterium"), "#888888")


def plot_all(names, seqs, gc,
             fft_res, fractal_res, jaccard_mat, jaccard_k,
             lz_res, wavelet_res, homology_res, mfdfa_res,
             svd_res, fisher_centroid, sheaf_res, gf4_res,
             dinu_res, kmer_ent_res, gc_skew_res,
             save_dir="."):
    os.makedirs(save_dir, exist_ok=True)
    colors = [_org_color(n) for n in names]

    # Helper: bar chart with organism-type colouring
    def bar(ax, vals, title, ylabel, hline=None, hline_label=None):
        bars = ax.bar(names, vals, color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
        if hline is not None:
            ax.axhline(hline, color="red", linestyle="--",
                       linewidth=1.0, label=hline_label or "")
            if hline_label:
                ax.legend(fontsize=7)
        return bars

    # -- 5.1  FFT frequencies
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    bar(axes[0],
        [fft_res[n]["dominant_freq"] for n in names],
        "Dominant FFT Frequency per Genome", "Frequency (cycles/bp)",
        hline=1/3, hline_label="1/3 = codon periodicity")
    bar(axes[1],
        [fft_res[n]["codon_power_ratio"] for n in names],
        "Codon-Band Power Ratio (±0.02 around 1/3 Hz)", "Power fraction")
    _add_legend(axes[0])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fft_analysis.png"), dpi=150)
    plt.close()
    log.info("   Saved fft_analysis.png")

    # -- 5.2  Fractal dimensions
    fig, ax = plt.subplots(figsize=(13, 4))
    bar(ax, [fractal_res.get(n, float("nan")) for n in names],
        "Chaos Walk Fractal Dimension (box-counting)", "D")
    _add_legend(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fractal_dimensions.png"), dpi=150)
    plt.close()
    log.info("   Saved fractal_dimensions.png")

    # -- 5.3  Jaccard heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(jaccard_mat, xticklabels=names, yticklabels=names,
                cmap="viridis", vmin=0, vmax=1, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    ax.set_title(f"Pairwise K-mer Jaccard Similarity (k={jaccard_k})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "jaccard_heatmap.png"), dpi=150)
    plt.close()
    log.info("   Saved jaccard_heatmap.png")

    # -- 5.4  LZ complexity
    fig, ax = plt.subplots(figsize=(13, 4))
    bar(ax, [lz_res.get(n, float("nan")) for n in names],
        "LZ Complexity (zlib compression ratio; lower = more structured)",
        "Compression ratio")
    _add_legend(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "lz_complexity.png"), dpi=150)
    plt.close()
    log.info("   Saved lz_complexity.png")

    # -- 5.5  Wavelet dominant scale
    fig, ax = plt.subplots(figsize=(13, 4))
    bar(ax, [wavelet_res[n]["dominant_scale_bp"] for n in names if n in wavelet_res],
        "Wavelet: Dominant Scale (bp)", "Scale (bp)")
    _add_legend(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "wavelet_scales.png"), dpi=150)
    plt.close()
    log.info("   Saved wavelet_scales.png")

    # -- 5.6  MFDFA scatter
    mf_names = [n for n in names if n in mfdfa_res
                and mfdfa_res[n].get("alpha_mean") == mfdfa_res[n].get("alpha_mean")]
    if mf_names:
        fig, ax = plt.subplots(figsize=(9, 6))
        for n in mf_names:
            ax.scatter(mfdfa_res[n]["alpha_mean"], mfdfa_res[n]["spectrum_width"],
                       c=_org_color(n), s=80, zorder=3)
            ax.annotate(n, (mfdfa_res[n]["alpha_mean"],
                            mfdfa_res[n]["spectrum_width"]),
                        fontsize=6, ha="left", va="bottom")
        ax.set_xlabel("Mean Hölder exponent α")
        ax.set_ylabel("Singularity spectrum width Δα")
        ax.set_title("Multifractal MFDFA — Singularity Spectrum")
        _add_legend(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "mfdfa_spectrum.png"), dpi=150)
        plt.close()
        log.info("   Saved mfdfa_spectrum.png")

    # -- 5.7  SVD entropy ratio
    fig, ax = plt.subplots(figsize=(13, 4))
    bar(ax, [svd_res.get(n, float("nan")) for n in names],
        "SVD Spectral Entropy Ratio (1.0 = white noise, lower = more structured)",
        "Entropy ratio")
    _add_legend(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "svd_entropy.png"), dpi=150)
    plt.close()
    log.info("   Saved svd_entropy.png")

    # -- 5.8  Dinucleotide ρ deviation
    fig, ax = plt.subplots(figsize=(13, 4))
    bar(ax, [dinu_res[n]["rho_mean_deviation"] for n in names if n in dinu_res],
        "Dinucleotide Relative Abundance Deviation (ρ from 1.0)",
        "Mean |ρ − 1|")
    _add_legend(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dinucleotide_rho.png"), dpi=150)
    plt.close()
    log.info("   Saved dinucleotide_rho.png")

    # -- 5.9  k-mer normalised entropy
    fig, ax = plt.subplots(figsize=(13, 4))
    bar(ax, [kmer_ent_res[n]["normalised_entropy"] for n in names if n in kmer_ent_res],
        "K-mer Normalised Shannon Entropy (H / H_max; 1.0 = uniform)",
        "H_norm")
    ax.set_ylim(0, 1.05)
    _add_legend(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "kmer_entropy.png"), dpi=150)
    plt.close()
    log.info("   Saved kmer_entropy.png")

    # -- 5.10  Sheaf H¹
    fig, ax = plt.subplots(figsize=(13, 4))
    bar(ax, [sheaf_res.get(n, float("nan")) for n in names],
        "Sheaf H¹ Obstruction (0 = perfectly consistent patches)",
        "Mean cocycle violation")
    _add_legend(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "sheaf_h1.png"), dpi=150)
    plt.close()
    log.info("   Saved sheaf_h1.png")

    # -- 5.11  GC-skew
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    bar(axes[0], [gc_skew_res[n]["gc_skew_mean"] for n in names if n in gc_skew_res],
        "GC-Skew Mean", "GC-skew")
    bar(axes[1], [gc_skew_res[n]["gc_skew_max_abs"] for n in names if n in gc_skew_res],
        "GC-Skew Max |deviation|", "max |GC-skew|")
    for ax in axes:
        _add_legend(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gc_skew.png"), dpi=150)
    plt.close()
    log.info("   Saved gc_skew.png")

    # -- 5.12  Fisher-Rao centroid distances
    fig, ax = plt.subplots(figsize=(13, 4))
    bar(ax, [fisher_centroid.get(n, float("nan")) for n in names],
        "Fisher-Rao Distance to Composition Centroid",
        "Geodesic distance (radians)")
    _add_legend(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fisher_rao_centroid.png"), dpi=150)
    plt.close()
    log.info("   Saved fisher_rao_centroid.png")

    # -- 5.13  Feature correlation matrix
    _plot_feature_correlation(names, fft_res, fractal_res, lz_res, svd_res,
                               sheaf_res, dinu_res, kmer_ent_res, gc_skew_res,
                               fisher_centroid, gc, save_dir)

    # -- 5.14  Radar chart
    _plot_radar(names, fft_res, fractal_res, lz_res, svd_res, sheaf_res,
                dinu_res, kmer_ent_res, gc, save_dir)

    # -- 5.15  Interactive de Bruijn (Plotly)
    G = build_de_bruijn_weighted(names, seqs, k=4)
    de_bruijn_analysis(G, save_dir)
    if G.number_of_nodes() <= 500:
        pos = nx.spring_layout(G, seed=42, k=0.5)
        ex, ey = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            ex.extend([x0, x1, None]); ey.extend([y0, y1, None])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ex, y=ey,
                                  line=dict(width=0.5, color="#aaa"),
                                  mode="lines", hoverinfo="none"))
        fig2.add_trace(go.Scatter(
            x=[pos[n][0] for n in G.nodes()],
            y=[pos[n][1] for n in G.nodes()],
            mode="markers+text", text=list(G.nodes()),
            marker=dict(size=6, color="tomato"),
            textposition="top center"))
        fig2.update_layout(title="de Bruijn Graph (k=4)", showlegend=False, height=700)
        fig2.write_html(os.path.join(save_dir, "de_bruijn.html"))
        log.info("   Saved de_bruijn.html")


def _add_legend(ax):
    """Add organism-type colour legend to axes."""
    seen_types = {ORGANISM_TYPES.get(n, "unknown") for n in ORGANISM_TYPES}
    patches = [mpatches.Patch(color=TYPE_COLORS[t], label=t)
               for t in seen_types if t in TYPE_COLORS]
    ax.legend(handles=patches, fontsize=6, loc="upper right",
              framealpha=0.5, ncol=2)


def _plot_feature_correlation(names, fft_res, fractal_res, lz_res, svd_res,
                               sheaf_res, dinu_res, kmer_ent_res, gc_skew_res,
                               fisher_centroid, gc, save_dir):
    """Correlation matrix of all scalar features."""
    rows = {}
    for n in names:
        rows[n] = [
            fft_res[n]["dominant_freq"]          if n in fft_res  else float("nan"),
            fft_res[n]["codon_power_ratio"]       if n in fft_res  else float("nan"),
            fractal_res.get(n, float("nan")),
            lz_res.get(n, float("nan")),
            svd_res.get(n, float("nan")),
            sheaf_res.get(n, float("nan")),
            dinu_res[n]["rho_mean_deviation"]     if n in dinu_res else float("nan"),
            kmer_ent_res[n]["normalised_entropy"] if n in kmer_ent_res else float("nan"),
            gc_skew_res[n]["gc_skew_mean"]        if n in gc_skew_res else float("nan"),
            fisher_centroid.get(n, float("nan")),
            float(gc.get(n, float("nan"))),
        ]
    feat_names = ["FFT_freq", "Codon_ratio", "Fractal_D", "LZ_ratio",
                  "SVD_entropy", "Sheaf_H1", "Dinu_rho", "Kmer_H_norm",
                  "GC_skew", "FisherRao_dist", "GC%"]
    mat = np.array(list(rows.values()))
    corr = np.full((len(feat_names), len(feat_names)), float("nan"))
    for i in range(len(feat_names)):
        for j in range(len(feat_names)):
            a, b = mat[:, i], mat[:, j]
            ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() > 3:
                corr[i, j] = pearsonr(a[ok], b[ok])[0]
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.isnan(corr)
    sns.heatmap(corr, xticklabels=feat_names, yticklabels=feat_names,
                cmap="RdBu_r", vmin=-1, vmax=1, center=0,
                annot=True, fmt=".2f", mask=mask, ax=ax, linewidths=0.3,
                annot_kws={"size": 7})
    ax.set_title("Feature Correlation Matrix (Pearson r)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_correlations.png"), dpi=150)
    plt.close()
    log.info("   Saved feature_correlations.png")


def _plot_radar(names, fft_res, fractal_res, lz_res, svd_res,
                sheaf_res, dinu_res, kmer_ent_res, gc, save_dir):
    """
    Radar (spider) chart of normalised scalar features per organism type.
    Plots the mean value per type — useful for high-level type comparison.
    """
    feat_labels = ["Codon\nratio", "Fractal D", "LZ\n(inv)",
                   "SVD\nentropy", "Sheaf H¹\n(inv)", "Dinu ρ", "Kmer H"]
    type_feats: Dict[str, List[List[float]]] = defaultdict(list)
    for n in names:
        otype = ORGANISM_TYPES.get(n, "bacterium")
        vec = [
            fft_res[n]["codon_power_ratio"] if n in fft_res else float("nan"),
            fractal_res.get(n, float("nan")),
            1 - lz_res.get(n, 0.5),              # invert: lower LZ = higher structure
            svd_res.get(n, float("nan")),
            1 - sheaf_res.get(n, 0.5),           # invert
            dinu_res[n]["rho_mean_deviation"]     if n in dinu_res    else float("nan"),
            kmer_ent_res[n]["normalised_entropy"] if n in kmer_ent_res else float("nan"),
        ]
        if not any(v != v for v in vec):          # skip if any nan
            type_feats[otype].append(vec)

    # Normalise each feature column to [0,1] across all values
    all_vals = [v for vecs in type_feats.values() for v in vecs]
    if not all_vals:
        return
    all_arr = np.array(all_vals)
    mins = np.nanmin(all_arr, 0)
    maxs = np.nanmax(all_arr, 0)
    rng  = maxs - mins + 1e-10

    N     = len(feat_labels)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    theta = np.append(theta, theta[0])

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for otype, vecs in type_feats.items():
        mean_v = np.mean(vecs, axis=0)
        norm_v = (mean_v - mins) / rng
        vals   = np.append(norm_v, norm_v[0])
        ax.plot(theta, vals, color=TYPE_COLORS.get(otype, "#888"),
                linewidth=2, label=otype)
        ax.fill(theta, vals, color=TYPE_COLORS.get(otype, "#888"), alpha=0.15)

    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(feat_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Genomic Feature Radar by Organism Type", pad=20, fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "radar_by_type.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    log.info("   Saved radar_by_type.png")

# ══════════════════════════════════════════════════════════════
# 6.  OUTPUT — CSV / JSON SUMMARY
# ══════════════════════════════════════════════════════════════

def save_summary(names, seqs, gc,
                 fft_res, fractal_res, lz_res, wavelet_res,
                 mfdfa_res, svd_res, fisher_centroid, sheaf_res,
                 gf4_res, dinu_res, kmer_ent_res, gc_skew_res,
                 save_dir="."):
    """Write summary_results.csv and summary_results.json."""
    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for name, seq in zip(names, seqs):
        otype = ORGANISM_TYPES.get(name, "unknown")
        row = {
            "name":             name,
            "organism_type":    otype,
            "length_bp":        len(seq),
            "gc_pct":           gc.get(name, float("nan")),
            "fft_dom_freq":     fft_res.get(name, {}).get("dominant_freq",   float("nan")),
            "fft_codon_ratio":  fft_res.get(name, {}).get("codon_power_ratio", float("nan")),
            "fft_amp_ratio":    fft_res.get(name, {}).get("amplitude_ratio",  float("nan")),
            "fractal_dim":      fractal_res.get(name, float("nan")),
            "lz_ratio":         lz_res.get(name, float("nan")),
            "wavelet_dom_scale": wavelet_res.get(name, {}).get("dominant_scale_bp", float("nan")),
            "wavelet_codon":    wavelet_res.get(name, {}).get("codon_scale_ratio",  float("nan")),
            "mfdfa_alpha":      mfdfa_res.get(name, {}).get("alpha_mean",      float("nan")),
            "mfdfa_width":      mfdfa_res.get(name, {}).get("spectrum_width",  float("nan")),
            "svd_entropy_ratio": svd_res.get(name, float("nan")),
            "fisher_centroid":  fisher_centroid.get(name, float("nan")),
            "sheaf_h1":         sheaf_res.get(name, float("nan")),
            "gf4_entropy":      gf4_res.get(name, float("nan")),
            "dinu_rho_dev":     dinu_res.get(name, {}).get("rho_mean_deviation", float("nan")),
            "kmer_entropy_norm": kmer_ent_res.get(name, {}).get("normalised_entropy", float("nan")),
            "gc_skew_mean":     gc_skew_res.get(name, {}).get("gc_skew_mean",    float("nan")),
            "gc_skew_max_abs":  gc_skew_res.get(name, {}).get("gc_skew_max_abs", float("nan")),
        }
        rows.append(row)

    csv_path = os.path.join(save_dir, "summary_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"   Saved {csv_path}")

    json_path = os.path.join(save_dir, "summary_results.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, default=str)
    log.info(f"   Saved {json_path}")

    # Print summary table to console
    _sep("Summary Table")
    hdr = f"{'Name':22s} {'Type':10s} {'Len(kbp)':>9s} {'GC%':>6s} " \
          f"{'FractalD':>9s} {'LZ':>6s} {'SVD_ratio':>10s} " \
          f"{'Dinu_ρ':>8s} {'Kmer_H':>7s}"
    log.info(hdr)
    log.info("─" * len(hdr))
    for r in rows:
        log.info(
            f"{r['name']:22s} {r['organism_type']:10s} "
            f"{r['length_bp']/1000:9.1f} {r['gc_pct']:6.2f} "
            f"{r['fractal_dim']:9.4f} {r['lz_ratio']:6.4f} "
            f"{r['svd_entropy_ratio']:10.5f} "
            f"{r['dinu_rho_dev']:8.4f} {r['kmer_entropy_norm']:7.4f}"
        )
    return rows

# ══════════════════════════════════════════════════════════════
# 7.  MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════

def run_experiment(email: str, full: bool = False, extra: bool = False,
                   ultra: bool = False, save_dir: str = "results", jaccard_k: int = 7):
    t0 = time.time()
    _sep("DNA ALCHEMY FRAMEWORK v5.0")

    names, seqs = load_all_sequences(email, full=full, extra=extra, ultra=ultra)
    if not seqs:
        log.error("No sequences loaded. Exiting.")
        return

    total_mb = sum(len(s) for s in seqs) / 1_000_000
    log.info(f"   Dataset: {len(seqs)} genomes  •  {total_mb:.1f} MB total")

    # ── Run all analyses ──────────────────────────────────────
    fft_res       = fft_spectral_analysis(names, seqs)
    fractal_res   = chaos_walk_fractal(names, seqs)
    jaccard_mat   = kmer_jaccard(names, seqs, k=jaccard_k)
    lz_res        = lz_complexity(names, seqs)
    wavelet_res   = wavelet_analysis(names, seqs)
    homology_res  = persistent_homology(names, seqs)
    mfdfa_res     = multifractal_analysis(names, seqs)
    svd_res       = svd_entanglement(names, seqs)
    fr_dist, fisher_centroid = fisher_rao_geometry(names, seqs, k=3)
    sheaf_res     = sheaf_cohomology_approx(names, seqs)
    gf4_res       = gf4_degree_distribution(names, seqs)
    dinu_res      = dinucleotide_relative_abundance(names, seqs)
    kmer_ent_res  = kmer_entropy(names, seqs, k=4)
    gc_skew_res   = gc_skew_analysis(names, seqs)

    gc = gc_content_and_correlations(names, seqs, {
        "fractal_dim":  fractal_res,
        "lz_ratio":     lz_res,
        "svd_entropy":  svd_res,
        "sheaf_h1":     sheaf_res,
        "dinu_rho_dev": {n: dinu_res[n]["rho_mean_deviation"] for n in names if n in dinu_res},
        "kmer_H_norm":  {n: kmer_ent_res[n]["normalised_entropy"] for n in names if n in kmer_ent_res},
        "gc_skew_std":  {n: gc_skew_res[n]["gc_skew_std"] for n in names if n in gc_skew_res},
        "fft_codon":    {n: fft_res[n]["codon_power_ratio"] for n in names if n in fft_res},
    })

    # ── Visualisations ────────────────────────────────────────
    _sep(f"Generating visualisations → {save_dir}/")
    plot_all(names, seqs, gc,
             fft_res, fractal_res, jaccard_mat, jaccard_k,
             lz_res, wavelet_res, homology_res, mfdfa_res,
             svd_res, fisher_centroid, sheaf_res, gf4_res,
             dinu_res, kmer_ent_res, gc_skew_res,
             save_dir=save_dir)

    # ── Save structured output ────────────────────────────────
    _sep("Saving Results")
    save_summary(names, seqs, gc,
                 fft_res, fractal_res, lz_res, wavelet_res,
                 mfdfa_res, svd_res, fisher_centroid, sheaf_res,
                 gf4_res, dinu_res, kmer_ent_res, gc_skew_res,
                 save_dir=save_dir)

    elapsed = time.time() - t0
    _sep("Complete")
    log.info(f"   {len(seqs)} genomes  •  {total_mb:.1f} MB  •  {elapsed:.1f}s elapsed")
    log.info(f"   All outputs in ./{save_dir}/")


# ══════════════════════════════════════════════════════════════
# 8.  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DNA Alchemy Framework v5.0 — multi-metric genomic analysis")
    parser.add_argument("--email", default="juraj.chobot60@gmail.com",
                        help="NCBI Entrez email (required)")
    parser.add_argument("--full", action="store_true",
                        help="Add E. coli + B. subtilis (~10 MB)")
    parser.add_argument("--extra", action="store_true",
                        help="Add ~8 more genomes (~32 MB total)")
    parser.add_argument("--ultra", action="store_true",
                        help="Add ~470 MB more well-chosen genomes (~500 MB total)")
    parser.add_argument("--output", default="results",
                        help="Output directory")
    parser.add_argument("--jaccard-k", type=int, default=7,
                        help="k for Jaccard similarity")
    args = parser.parse_args()

    if args.email == "your@email.com":
        log.error("Please provide a real email with --email your@email.com")
        sys.exit(1)

    run_experiment(
        email=args.email,
        full=args.full,
        extra=args.extra,
        ultra=args.ultra,
        save_dir=args.output,
        jaccard_k=args.jaccard_k,
    )
