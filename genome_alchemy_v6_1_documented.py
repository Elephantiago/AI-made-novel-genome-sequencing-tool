#!/usr/bin/env python3
"""
Genome Alchemy v6.1 Taxonomy-Aware Parallel Framework
=====================================================

This file is a fully documented version of the taxonomy-aware, parallel Genome
Alchemy pipeline. It keeps the bold spirit of the original project while making
the implementation easier to audit, extend, and discuss on GitHub.

What this script does
---------------------
1. Downloads or loads representative genomic sequences.
2. Cleans and labels them with an explicit organism taxonomy.
3. Slices each genome into reproducible analysis windows.
4. Generates matched null-control windows (shuffle / Markov / reverse-complement).
5. Extracts a broad panel of multiscale sequence features in parallel.
6. Compares observed windows to controls using effect sizes and empirical tests.
7. Writes reproducible CSV / JSON artifacts plus optional figures.
8. Optionally performs de Bruijn graph diagnostics on the full sequences.

Design principles
-----------------
- **Falsifiable**: controls are treated as first-class citizens.
- **Windowed**: metrics are computed on multiple windows, not a single prefix.
- **Parallel**: work is distributed across CPU cores using worker processes.
- **Taxonomy-aware**: humans, plants, fungi, invertebrates, etc. are not
  collapsed into “bacterium”.
- **Auditable**: seeds, manifests, and stable identifiers make reruns traceable.

Practical note
--------------
This code is intentionally verbose. The extra comments and docstrings are there
for maintainability and for future contributors who may not be familiar with
signal processing, comparative genomics, or scientific Python tooling.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time

# Keep native BLAS/NumPy from oversubscribing every worker process.
# Feature extraction is parallelized at the Python process level below.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Reference accession tiers and taxonomy tables
# ---------------------------------------------------------------------------

# Heavy optional libraries are imported lazily. This matters because worker
# processes should not waste seconds importing plotting/graph packages they do
# not use for per-window feature extraction.
plt = None
nx = None
pearsonr = None

DNA = "ACGT"
DNA_SET = set(DNA)
COMP = str.maketrans("ACGTUNacgtun", "TGCAANtgcaan")
BASE_TO_INT = np.full(256, 4, dtype=np.uint8)
for i, b in enumerate(b"ACGT"):
    BASE_TO_INT[b] = i
BASE_COMPLEX = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex128)
BASE_PURINE = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)  # A,C,G,T
BASE_KETO = np.array([-1.0, 1.0, 1.0, -1.0], dtype=np.float64)   # A,C,G,T

ACCESSIONS_CORE = {
    "phiX174": "NC_001422",
    "pUC19": "L09137",
    "SARS-CoV-2": "NC_045512",
    "M13": "NC_003287",
    "pBR322": "J01749",
    "SV40": "NC_001669",
    "Lambda": "NC_001416",
    "PSTVd": "NC_002030",
    "Human_mtDNA": "NC_012920",
    "T7_phage": "NC_001604",
    "MS2_phage": "NC_001417",
}
ACCESSIONS_FULL = {
    "E_coli_K12": "NC_000913",
    "B_subtilis": "NC_000964",
}
ACCESSIONS_EXTRA = {
    "P_aeruginosa_PA_O1": "NC_002516",
    "M_tuberculosis_H37Rv": "NC_000962",
    "V_cholerae_chr1": "NC_002505",
    "S_pneumoniae_R6": "NC_003098",
    "H_influenzae_RdKW20": "NC_000907",
    "M_jannaschii": "NC_000916",
    "T_thermophilus": "NC_006461",
    "S_cerevisiae_chrIV": "NC_001136",
}
ACCESSIONS_ULTRA = {
    "Salmonella_enterica": "NC_003197",
    "Streptomyces_coelicolor": "NC_005363",
    "Pseudomonas_putida": "NC_009512",
    "Bacillus_cereus": "NC_003909",
    "Rhodobacter_sphaeroides": "NC_007779",
    "Sulfolobus_solfataricus": "NC_002754",
    "Arabidopsis_chr1": "NC_003070",
    # NC_004353 is Drosophila chromosome 4, not chromosome 2. The old label was misleading.
    "Drosophila_chr4": "NC_004353",
    "C_elegans_chrI": "NC_003279",
    "S_cerevisiae_full": "NC_001143",
    "Human_chr1": "NC_000001",
    "Human_chr2": "NC_000002",
    "Human_chr19": "NC_000019",
    "Human_chr21": "NC_000021",
}
ACCESSIONS_DIVERSE_EUKARYOTES = {
    # Plant: complete the small Arabidopsis chromosome set beyond chr1 (~89 Mb)
    "Arabidopsis_chr2": "NC_003071",
    "Arabidopsis_chr3": "NC_003074",
    "Arabidopsis_chr4": "NC_003075",
    "Arabidopsis_chr5": "NC_003076",
    # Simple animal: complete C. elegans beyond chrI (~85 Mb)
    "C_elegans_chrII": "NC_003280",
    "C_elegans_chrIII": "NC_003281",
    "C_elegans_chrIV": "NC_003282",
    "C_elegans_chrV": "NC_003283",
    "C_elegans_chrX": "NC_003284",
    # Simple animal / insect: Drosophila X plus existing chr4 (~23.5 Mb new)
    "Drosophila_chrX": "NC_004354",
    # Extra human chromosomes that keep the expansion near the 400 Mb target (~207 Mb)
    "Human_chr22": "NC_000022",
    "Human_chrX": "NC_000023",
}

ORGANISM_TYPES = {
    "phiX174": "phage", "M13": "phage", "Lambda": "phage", "T7_phage": "phage", "MS2_phage": "phage",
    "pUC19": "plasmid", "pBR322": "plasmid",
    "SARS-CoV-2": "virus", "SV40": "virus", "PSTVd": "viroid", "Human_mtDNA": "organelle",
    "E_coli_K12": "bacterium", "B_subtilis": "bacterium", "P_aeruginosa_PA_O1": "bacterium",
    "M_tuberculosis_H37Rv": "bacterium", "V_cholerae_chr1": "bacterium", "S_pneumoniae_R6": "bacterium",
    "H_influenzae_RdKW20": "bacterium", "T_thermophilus": "bacterium", "Salmonella_enterica": "bacterium",
    "Streptomyces_coelicolor": "bacterium", "Pseudomonas_putida": "bacterium", "Bacillus_cereus": "bacterium",
    "Rhodobacter_sphaeroides": "bacterium",
    "M_jannaschii": "archaeon", "Sulfolobus_solfataricus": "archaeon",
    "S_cerevisiae_chrIV": "fungus", "S_cerevisiae_full": "fungus",
    "Arabidopsis_chr1": "plant", "Arabidopsis_chr2": "plant", "Arabidopsis_chr3": "plant",
    "Arabidopsis_chr4": "plant", "Arabidopsis_chr5": "plant",
    "Drosophila_chr4": "animal_invertebrate", "Drosophila_chrX": "animal_invertebrate",
    "C_elegans_chrI": "animal_invertebrate", "C_elegans_chrII": "animal_invertebrate",
    "C_elegans_chrIII": "animal_invertebrate", "C_elegans_chrIV": "animal_invertebrate",
    "C_elegans_chrV": "animal_invertebrate", "C_elegans_chrX": "animal_invertebrate",
    "Human_chr1": "human", "Human_chr2": "human", "Human_chr19": "human", "Human_chr21": "human",
    "Human_chr22": "human", "Human_chrX": "human",
}
TYPE_COLORS = {
    "phage": "#e41a1c", "virus": "#ff7f00", "viroid": "#f781bf", "plasmid": "#a65628",
    "organelle": "#888888", "bacterium": "#377eb8", "archaeon": "#984ea3",
    "fungus": "#8c564b", "plant": "#2ca02c", "animal_invertebrate": "#17becf", "animal_vertebrate": "#1f77b4",
    "human": "#d62728", "eukaryote": "#4daf4a", "unknown": "#777777", "control": "#222222",
}
TAXONOMY_ALIASES = {
    "eukaryote": "eukaryote", "eukaryota": "eukaryote",
    "animal": "animal_vertebrate", "metazoan": "animal_vertebrate",
    "invertebrate": "animal_invertebrate", "vertebrate": "animal_vertebrate",
    "mammal": "animal_vertebrate", "human": "human", "plant": "plant", "fungus": "fungus", "fungi": "fungus",
}

log = logging.getLogger("dna_alchemy_v6")


@dataclass(frozen=True)
class SequenceRecord:
    name: str
    accession: str
    organism_type: str
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode()).hexdigest()


@dataclass(frozen=True)
class WindowRecord:
    organism: str
    accession: str
    organism_type: str
    source: str              # observed, shuffle, markov0, markov1, revcomp
    replicate: int
    start: int
    end: int
    sequence: str

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass
class Config:
    out: Path
    cache: Path
    email: str = "anonymous@example.com"
    seed: int = 42
    window: int = 200_000
    windows_per_genome: int = 8
    min_window: int = 1_000
    jaccard_k: int = 7
    graph_k: int = 4
    controls: List[str] = field(default_factory=lambda: ["shuffle", "markov1", "revcomp"])
    make_plots: bool = True
    fetch: bool = True
    workers: int = 0
    chunksize: int = 4
    progress_every: int = 25
    debruijn: bool = True


# ----------------------------- IO -----------------------------------------

def setup_logging(out: Path) -> None:
    """Configure console/file logging for the current run.

    A dedicated log file is written inside the output directory, while a matching
    stream handler prints progress to the console. Having both makes long runs much
    easier to monitor and later debug."""
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log", mode="w")],
    )


def clean_sequence(seq: str) -> str:
    """Normalize a raw nucleotide string.

    Only canonical DNA bases A/C/G/T are retained. U is converted to T before the
    filtering step. This keeps downstream numerical encodings simple and avoids
    silent propagation of ambiguous symbols."""
    seq = seq.upper().replace("U", "T")
    return "".join(ch for ch in seq if ch in DNA_SET)


def read_fasta(path: Path, name: Optional[str] = None) -> SequenceRecord:
    """Read a local FASTA file into a :class:`SequenceRecord`.

    The function supports plain-text or gzip-compressed FASTA files via
    :func:`open_maybe_gzip`. The record name is taken from the explicit `name`
    argument when provided, otherwise from the FASTA header or file stem."""
    label = name or path.stem
    parts: List[str] = []
    first_header: Optional[str] = None
    with open_maybe_gzip(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if first_header is None:
                    first_header = line[1:].strip()
                continue
            parts.append(line.strip())
    if name is None and first_header:
        # Use a compact, filesystem-safe version of the FASTA header when it
        # carries useful taxonomy hints such as "Homo sapiens" or "Arabidopsis".
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", first_header)[:80] or path.stem
    seq = clean_sequence("".join(parts))
    return SequenceRecord(label, path.name, infer_organism_type(label), seq)


def open_maybe_gzip(path: Path, mode: str):
    """Open a text file that may or may not be gzip-compressed."""
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode)


def fetch_ncbi(name: str, accession: str, cfg: Config) -> Optional[SequenceRecord]:
    """Fetch a genome from NCBI and cache it locally.

    Caching is essential because many experimental reruns reuse the same accessions.
    The function therefore prefers the cached FASTA when available and falls back to
    NCBI Entrez only when needed."""
    cfg.cache.mkdir(parents=True, exist_ok=True)
    cache_path = cfg.cache / f"{accession}.fasta"
    if not cache_path.exists():
        if not cfg.fetch:
            log.warning("cache miss for %s/%s and --no-fetch was used", name, accession)
            return None
        try:
            from Bio import Entrez
            Entrez.email = cfg.email
            log.info("fetching %-24s %s", name, accession)
            handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
            text = handle.read()
            handle.close()
            cache_path.write_text(text)
            time.sleep(0.35)
        except Exception as exc:
            log.error("failed to fetch %s %s: %s", name, accession, exc)
            return None
    return read_fasta(cache_path, name=name)


def accession_set(args: argparse.Namespace) -> Dict[str, str]:
    # Tier semantics are cumulative and intentionally explicit:
    #   core  -> small replicons/viruses/phages
    #   full  -> core + first two bacterial chromosomes
    #   extra -> full + broader prokaryotes + one yeast chromosome
    #   ultra -> extra + original large/eukaryote panel
    #   mega  -> ultra + ~400 Mb diverse eukaryote expansion
    """Assemble the accession dictionary implied by the CLI flags.

    The tier flags are cumulative by design:
    `--full` ⊂ `--extra` ⊂ `--ultra`, while `--diverse-eukaryotes` adds the broader
    eukaryotic expansion and `--mega` enables everything."""
    acc = dict(ACCESSIONS_CORE)
    if args.full or args.extra or args.ultra or args.mega:
        acc.update(ACCESSIONS_FULL)
    if args.extra or args.ultra or args.mega:
        acc.update(ACCESSIONS_EXTRA)
    if args.ultra or args.mega:
        acc.update(ACCESSIONS_ULTRA)
    if args.diverse_eukaryotes or args.mega:
        acc.update(ACCESSIONS_DIVERSE_EUKARYOTES)
    return acc


def infer_organism_type(name: str, accession: str = "", explicit: str = "unknown") -> str:
    """Classify bundled and user FASTA records without defaulting unknowns to bacteria."""
    if explicit and explicit != "unknown":
        return TAXONOMY_ALIASES.get(explicit.lower(), explicit)
    if name in ORGANISM_TYPES:
        return ORGANISM_TYPES[name]
    n = name.lower()
    rules = [
        (("phage", "lambda", "phix", "phi"), "phage"),
        (("plasmid", "puc", "pbr"), "plasmid"),
        (("viroid", "pstvd"), "viroid"),
        (("mitochond", "mtdna", "chloroplast", "plastid"), "organelle"),
        (("virus", "sars", "cov", "sv40"), "virus"),
        (("archae", "sulfolobus", "jannaschii"), "archaeon"),
        (("arabidopsis", "oryza", "rice", "zea", "maize", "plant"), "plant"),
        (("saccharomyces", "cerevisiae", "yeast", "fung"), "fungus"),
        (("caenorhabditis", "elegans", "drosophila", "fly", "invertebrate"), "animal_invertebrate"),
        (("human", "homo_sapiens", "homo sapiens", "chr1", "chrx", "chry"), "human"),
        (("bacter", "ecoli", "e_coli", "subtilis", "pseudomonas", "streptomyces", "salmonella",
          "tuberculosis", "thermophilus", "cholerae", "pneumoniae", "influenzae"), "bacterium"),
    ]
    for needles, typ in rules:
        if any(x in n for x in needles):
            return typ
    return "unknown"


# -------------------------- windowing & controls ---------------------------

def deterministic_windows(seq_len: int, window: int, n: int, min_window: int, rng: random.Random) -> List[Tuple[int, int]]:
    """Create reproducible analysis windows for one sequence.

    Windows are sampled using a deterministic RNG so that repeated runs with the
    same seed revisit the same genomic regions. This makes comparisons across code
    versions and hardware straightforward."""
    if seq_len < min_window:
        return [(0, seq_len)] if seq_len else []
    w = min(window, seq_len)
    if seq_len <= w:
        return [(0, seq_len)]
    starts = {0, seq_len - w}
    if n <= 2:
        return sorted((s, s + w) for s in list(starts)[:n])
    # Mix quantiles with random windows: reproducible coverage without pure prefix bias.
    for q in np.linspace(0, 1, max(0, n - 2)):
        starts.add(int(round(q * (seq_len - w))))
    while len(starts) < n:
        starts.add(rng.randint(0, seq_len - w))
    return sorted((s, s + w) for s in starts)[:n]


def make_controls(seq: str, controls: Sequence[str], rng: random.Random) -> Dict[str, str]:
    """Generate null-control sequences matched to an observed window.

    Each requested control is derived from the same observed window, ensuring that
    observed vs. null comparisons are local and fair. Controls are intentionally
    simple but complementary: random shuffles destroy order, Markov models preserve
    increasingly more local structure, and reverse complement preserves base content
    while altering strand orientation."""
    out: Dict[str, str] = {}
    for control in controls:
        c = control.strip().lower()
        if c == "shuffle":
            chars = list(seq)
            rng.shuffle(chars)
            out[c] = "".join(chars)
        elif c == "markov0":
            out[c] = markov0(seq, rng)
        elif c == "markov1":
            out[c] = markov1(seq, rng)
        elif c == "revcomp":
            out[c] = seq.translate(COMP)[::-1].upper()
        elif c in ("", "none", "observed"):
            continue
        else:
            raise ValueError(f"unknown control: {control}")
    return out


def markov0(seq: str, rng: random.Random) -> str:
    """Sample an i.i.d. sequence from the mononucleotide composition."""
    counts = Counter(seq)
    total = sum(counts.values()) or 1
    alphabet = list(DNA)
    probs = [counts[b] / total for b in alphabet]
    return "".join(rng.choices(alphabet, weights=probs, k=len(seq)))


def markov1(seq: str, rng: random.Random) -> str:
    """Sample a first-order Markov surrogate preserving dinucleotide flow."""
    if len(seq) < 2:
        return seq
    transitions: Dict[str, Counter] = {b: Counter() for b in DNA}
    starts = Counter(seq)
    for a, b in zip(seq[:-1], seq[1:]):
        if a in DNA_SET and b in DNA_SET:
            transitions[a][b] += 1
    cur = rng.choices(list(DNA), weights=[starts[b] + 1 for b in DNA], k=1)[0]
    chars = [cur]
    for _ in range(len(seq) - 1):
        weights = [transitions[cur][b] + 1 for b in DNA]
        cur = rng.choices(list(DNA), weights=weights, k=1)[0]
        chars.append(cur)
    return "".join(chars)


def stable_seed(*parts: object) -> int:
    """Stable 32-bit seed; unlike Python hash(), this is reproducible across runs."""
    text = "|".join(str(p) for p in parts)
    return int(hashlib.blake2b(text.encode(), digest_size=8).hexdigest(), 16) & 0xFFFFFFFF


def iter_windows(records: Sequence[SequenceRecord], cfg: Config) -> Iterator[WindowRecord]:
    """Yield observed and control windows for every input genome."""
    rng = random.Random(cfg.seed)
    for rec in records:
        spans = deterministic_windows(rec.length, cfg.window, cfg.windows_per_genome, cfg.min_window, rng)
        for rep, (start, end) in enumerate(spans):
            seq = rec.sequence[start:end]
            yield WindowRecord(rec.name, rec.accession, rec.organism_type, "observed", rep, start, end, seq)
            control_rng = random.Random(stable_seed(cfg.seed, rec.name, rep))
            for source, cseq in make_controls(seq, cfg.controls, control_rng).items():
                yield WindowRecord(rec.name, rec.accession, rec.organism_type, source, rep, start, end, cseq)


# ------------------------------- features ----------------------------------

def encode_uint(seq: str) -> np.ndarray:
    """Encode DNA bases as compact integers 0..3 suitable for NumPy math."""
    raw = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
    encoded = BASE_TO_INT[raw]
    return encoded[encoded < 4]


def base_counts(seq: str) -> Dict[str, int]:
    """Count canonical DNA bases in a sequence."""
    c = Counter(seq)
    return {b: int(c.get(b, 0)) for b in DNA}


def gc_pct(seq: str) -> float:
    """Return GC percentage for a sequence."""
    return 100.0 * (seq.count("G") + seq.count("C")) / max(len(seq), 1)


def shannon_entropy_from_counts(counts: Sequence[int]) -> float:
    """Compute Shannon entropy from discrete event counts."""
    arr = np.asarray(counts, dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return float("nan")
    p = arr[arr > 0] / total
    return float(-(p * np.log2(p)).sum())


def kmer_counts(seq: str, k: int) -> Counter:
    """Count all canonical k-mers in a sequence."""
    if len(seq) < k:
        return Counter()
    return Counter(seq[i:i+k] for i in range(len(seq) - k + 1) if set(seq[i:i+k]) <= DNA_SET)


def kmer_entropy_norm(seq: str, k: int) -> float:
    """Compute normalized Shannon entropy of the k-mer spectrum.

    The result is divided by the maximum possible entropy for the occupied alphabet,
    placing different sequences on a more comparable 0..1-like scale."""
    cnt = kmer_counts(seq, k)
    if not cnt:
        return float("nan")
    return shannon_entropy_from_counts(list(cnt.values())) / (2 * k)


def kmer_occupancy(seq: str, k: int) -> float:
    """Return the fraction of the full k-mer space observed at least once."""
    return len(kmer_counts(seq, k)) / float(4 ** k)


def dinucleotide_rho_deviation(seq: str) -> float:
    """Measure deviation from mononucleotide-expected dinucleotide usage.

    This is a compact genomic-signature statistic: values near zero indicate that
    dinucleotide frequencies are close to the product of single-base frequencies,
    whereas larger values indicate stronger context dependence."""
    if len(seq) < 2:
        return float("nan")
    mono = Counter(seq)
    di = kmer_counts(seq, 2)
    n = len(seq)
    devs = []
    for a in DNA:
        for b in DNA:
            expected = (mono[a] / n) * (mono[b] / n)
            observed = di[a+b] / max(n - 1, 1)
            if expected > 0:
                devs.append(abs(observed / expected - 1.0))
    return float(np.mean(devs)) if devs else float("nan")


def fft_codon_metrics(seq: str) -> Dict[str, float]:
    """Extract codon-periodicity metrics from a complex-valued DNA encoding.

    The FFT is used here as a fast detector of periodic structure, especially the
    classic period-3 / frequency-1/3 signature associated with coding structure."""
    arr = encode_uint(seq)
    if len(arr) < 32:
        return nan_dict("fft_peak_freq", "fft_peak_ratio", "fft_codon_band", "fft_codon_z")
    signal = BASE_COMPLEX[arr]
    signal = signal - signal.mean()
    # Complex DNA encoding requires full FFT, not rFFT.
    fft = np.fft.fft(signal)
    half = len(fft) // 2
    power = np.abs(fft[:half]) ** 2
    freqs = np.fft.fftfreq(len(signal), d=1.0)[:half]
    if len(power) <= 2:
        return nan_dict("fft_peak_freq", "fft_peak_ratio", "fft_codon_band", "fft_codon_z")
    power[0] = 0
    peak = int(np.argmax(power))
    band = np.abs(freqs - 1 / 3) < 0.015
    bg = power[~band]
    codon_power = float(power[band].sum() / (power.sum() + 1e-15))
    codon_z = float((power[band].mean() - bg.mean()) / (bg.std() + 1e-15)) if band.any() and len(bg) > 2 else float("nan")
    return {
        "fft_peak_freq": float(freqs[peak]),
        "fft_peak_ratio": float(power[peak] / (power.mean() + 1e-15)),
        "fft_codon_band": codon_power,
        "fft_codon_z": codon_z,
    }


def wavelet_codon_proxy(seq: str) -> Dict[str, float]:
    # Fast, dependency-free local version: Haar-like variance of every 3rd residue phase.
    """Compute a lightweight wavelet-inspired codon periodicity proxy.

    This is not a full CWT implementation; it is a cheap, robust proxy intended for
    large comparative runs where runtime matters more than detailed scaleograms."""
    arr = encode_uint(seq)
    if len(arr) < 90:
        return nan_dict("phase3_strength", "phase3_entropy", "phase3_gc_delta")
    gc = np.isin(arr, [1, 2]).astype(float)
    phase_means = np.array([gc[i::3].mean() for i in range(3)], dtype=float)
    p = phase_means / (phase_means.sum() + 1e-15)
    phase_entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum() / math.log2(3))
    return {
        "phase3_strength": float(np.std(phase_means)),
        "phase3_entropy": phase_entropy,
        "phase3_gc_delta": float(phase_means.max() - phase_means.min()),
    }


def chaos_fractal_dim(seq: str) -> float:
    """Estimate a box-counting fractal dimension from a chaos walk.

    The sequence is mapped into a 2D walk based on purine/pyrimidine and keto/amino
    contrasts. The roughness of the resulting trajectory is summarized through an
    approximate fractal dimension."""
    arr = encode_uint(seq)
    if len(arr) < 500:
        return float("nan")
    x = np.cumsum(BASE_PURINE[arr])
    y = np.cumsum(BASE_KETO[arr])
    pts = np.column_stack([x, y]).astype(float)
    max_exp = max(4, int(np.log2(len(pts))) - 2)
    scales = 2.0 ** np.arange(3, max_exp)
    if len(scales) < 3:
        return float("nan")
    counts = []
    for s in scales:
        boxes = np.floor((pts - pts.min(axis=0)) / s).astype(np.int64)
        counts.append(len(np.unique(boxes, axis=0)))
    counts = np.asarray(counts, dtype=float)
    ok = counts > 1
    if ok.sum() < 3:
        return float("nan")
    slope = np.polyfit(np.log(1.0 / scales[ok]), np.log(counts[ok]), 1)[0]
    return float(slope)


def lz_zlib_ratio(seq: str) -> float:
    """Use zlib compression ratio as a coarse complexity proxy."""
    if not seq:
        return float("nan")
    import zlib
    return len(zlib.compress(seq.encode(), 9)) / len(seq.encode())


def gc_skew_metrics(seq: str, bins: int = 32) -> Dict[str, float]:
    """Summarize GC skew across bins along the window.

    This captures strand asymmetry and replication-associated biases in a simple,
    length-agnostic form."""
    if len(seq) < 1000:
        return nan_dict("gc_skew_mean", "gc_skew_std", "gc_skew_max_abs")
    w = max(100, len(seq) // bins)
    vals = []
    for i in range(0, len(seq) - w + 1, w):
        s = seq[i:i+w]
        g, c = s.count("G"), s.count("C")
        vals.append((g - c) / max(g + c, 1))
    arr = np.asarray(vals, dtype=float)
    return {
        "gc_skew_mean": float(arr.mean()),
        "gc_skew_std": float(arr.std()),
        "gc_skew_max_abs": float(np.max(np.abs(arr))),
    }


def svd_entropy_ratio(seq: str, width: int = 64) -> float:
    """Estimate a matrix-embedding spectral entropy ratio.

    A Hankel-like embedding of the sequence is decomposed with SVD. The normalized
    entropy of singular values provides a compact measure of structural richness."""
    arr = encode_uint(seq)
    n = (len(arr) // width) * width
    if n < width * 8:
        return float("nan")
    mat = arr[:n].reshape(-1, width).astype(float)
    mat -= mat.mean(axis=0, keepdims=True)
    try:
        s = np.linalg.svd(mat, compute_uv=False)
    except np.linalg.LinAlgError:
        return float("nan")
    p = (s ** 2) / ((s ** 2).sum() + 1e-15)
    ent = -(p[p > 0] * np.log2(p[p > 0])).sum()
    return float(ent / math.log2(len(p)))


def mfdfa_fast(seq: str) -> Dict[str, float]:
    # Compact MFDFA-like summary: not a full formal spectrum, but stable and cheap.
    """Return a fast approximation to multifractal DFA summary metrics.

    The implementation is intentionally simplified to remain practical for large,
    parallel runs. It yields summary values analogous to an average Hölder exponent
    and singularity-spectrum width."""
    arr = encode_uint(seq)
    if len(arr) < 4096:
        return nan_dict("mfdfa_alpha_mean", "mfdfa_width")
    x = BASE_PURINE[arr]
    y = np.cumsum(x - x.mean())
    scales = np.unique(np.logspace(np.log10(16), np.log10(max(32, len(y) // 8)), 10).astype(int))
    qs = np.array([-4, -2, 0, 2, 4], dtype=float)
    h = []
    for q in qs:
        Fs = []
        for s in scales:
            if s < 8 or len(y) // s < 4:
                continue
            segs = y[: (len(y)//s)*s].reshape(-1, s)
            t = np.arange(s)
            rms = []
            for seg in segs:
                coef = np.polyfit(t, seg, 1)
                trend = coef[0] * t + coef[1]
                rms.append(np.sqrt(np.mean((seg - trend) ** 2)) + 1e-12)
            rms = np.asarray(rms)
            if q == 0:
                Fs.append(np.exp(0.5 * np.mean(np.log(rms ** 2))))
            else:
                Fs.append((np.mean(rms ** q)) ** (1.0 / q))
        if len(Fs) >= 4:
            h.append(np.polyfit(np.log(scales[:len(Fs)]), np.log(Fs), 1)[0])
    if len(h) < 3:
        return nan_dict("mfdfa_alpha_mean", "mfdfa_width")
    h = np.asarray(h)
    return {"mfdfa_alpha_mean": float(h.mean()), "mfdfa_width": float(h.max() - h.min())}


def sheaf_patch_obstruction(seq: str, patch: int = 3000, k: int = 3) -> float:
    # Operationalized as dispersion of local k-mer distributions around the global distribution.
    """Approximate a patch-consistency / sheaf-obstruction score.

    The code does not implement full algebraic sheaf cohomology. Instead, it uses a
    practical patchwise inconsistency proxy that can still reveal whether local
    codon-usage neighborhoods fit together smoothly."""
    if len(seq) < patch * 2:
        patch = max(200, len(seq) // 4)
    if patch < k * 10:
        return float("nan")
    global_vec = kmer_vector(seq, k)
    dists = []
    for i in range(0, len(seq) - patch + 1, patch):
        local = kmer_vector(seq[i:i+patch], k)
        dists.append(hellinger(local, global_vec))
    return float(np.mean(dists)) if dists else float("nan")


def fisher_centroid_distance(seq: str, k: int = 2) -> float:
    # This field is later replaced by across-organism centroid when summarizing.
    """Compute a simple Fisher–Rao distance to a uniform centroid."""
    vec = kmer_vector(seq, k)
    uniform = np.full_like(vec, 1.0 / len(vec))
    return fisher_rao(vec, uniform)


def kmer_vector(seq: str, k: int) -> np.ndarray:
    """Convert a sequence into a probability vector over all canonical k-mers."""
    cnt = kmer_counts(seq, k)
    words = all_kmers(k)
    arr = np.array([cnt[w] for w in words], dtype=float)
    s = arr.sum()
    if s <= 0:
        return np.full(len(words), 1.0 / len(words))
    return arr / s


def all_kmers(k: int) -> List[str]:
    """Enumerate all canonical DNA k-mers in lexicographic order."""
    words = [""]
    for _ in range(k):
        words = [w + b for w in words for b in DNA]
    return words


def hellinger(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Hellinger distance between two probability vectors."""
    return float(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / math.sqrt(2))


def fisher_rao(p: np.ndarray, q: np.ndarray) -> float:
    """Compute the Fisher–Rao distance between discrete distributions."""
    p = p / (p.sum() + 1e-15)
    q = q / (q.sum() + 1e-15)
    inner = np.clip(np.sum(np.sqrt(p * q)), 0.0, 1.0)
    return float(2.0 * math.acos(inner))


def nan_dict(*keys: str) -> Dict[str, float]:
    """Convenience helper returning a dictionary filled with NaNs."""
    return {k: float("nan") for k in keys}


def compute_features(win: WindowRecord, cfg: Config) -> Dict[str, object]:
    """Compute the full feature panel for one :class:`WindowRecord`.

    This is the main per-window worker payload. It keeps the feature extraction
    logic in one place so that single-process and multi-process execution paths stay
    consistent."""
    seq = win.sequence
    feats: Dict[str, object] = {
        "organism": win.organism,
        "accession": win.accession,
        "organism_type": win.organism_type,
        "source": win.source,
        "replicate": win.replicate,
        "start": win.start,
        "end": win.end,
        "length_bp": len(seq),
        "gc_pct": gc_pct(seq),
        "a_count": seq.count("A"), "c_count": seq.count("C"),
        "g_count": seq.count("G"), "t_count": seq.count("T"),
        "lz_ratio": lz_zlib_ratio(seq),
        "fractal_dim": chaos_fractal_dim(seq),
        "dinu_rho_dev": dinucleotide_rho_deviation(seq),
        "kmer_entropy_norm": kmer_entropy_norm(seq, cfg.jaccard_k),
        "kmer_occupancy": kmer_occupancy(seq, cfg.jaccard_k),
        "svd_entropy_ratio": svd_entropy_ratio(seq),
        "sheaf_h1_proxy": sheaf_patch_obstruction(seq),
        "fisher_to_uniform": fisher_centroid_distance(seq, k=2),
        "debruijn_k4_occupancy": kmer_occupancy(seq, cfg.graph_k),
    }
    feats.update(fft_codon_metrics(seq))
    feats.update(wavelet_codon_proxy(seq))
    feats.update(gc_skew_metrics(seq))
    feats.update(mfdfa_fast(seq))
    return feats


def _compute_features_worker(job: Tuple[int, WindowRecord, Config]) -> Tuple[int, Dict[str, object]]:
    """Top-level multiprocessing worker. Returns the original index to preserve order."""
    idx, win, cfg = job
    return idx, compute_features(win, cfg)


def compute_all_features(windows: Sequence[WindowRecord], cfg: Config) -> List[Dict[str, object]]:
    """Compute window features using multiple CPU cores when requested."""
    total = len(windows)
    if total == 0:
        return []
    workers = cfg.workers
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 1) - 1)
    chunksize = max(1, cfg.chunksize)
    log.info("feature extraction plan: %d windows, workers=%d, chunksize=%d", total, workers, chunksize)
    rows: List[Optional[Dict[str, object]]] = [None] * total
    t0 = time.time()
    if workers == 1:
        for idx, win in enumerate(windows):
            if idx == 0 or (idx + 1) % cfg.progress_every == 0 or idx + 1 == total:
                log_progress(idx + 1, total, t0, win)
            try:
                rows[idx] = compute_features(win, cfg)
            except Exception as exc:
                log.exception("feature extraction failed for %s/%s/%s: %s", win.organism, win.source, win.replicate, exc)
        return [r for r in rows if r is not None]
    jobs = ((i, win, cfg) for i, win in enumerate(windows))
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for idx, row in ex.map(_compute_features_worker, jobs, chunksize=chunksize):
            rows[idx] = row
            completed += 1
            if completed == 1 or completed % cfg.progress_every == 0 or completed == total:
                log_progress(completed, total, t0, windows[idx])
    missing = [i for i, r in enumerate(rows) if r is None]
    if missing:
        raise RuntimeError(f"feature extraction produced {len(missing)} missing rows")
    return [r for r in rows if r is not None]


def log_progress(done: int, total: int, t0: float, win: WindowRecord) -> None:
    """Emit a human-friendly progress / ETA message during extraction."""
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else float("nan")
    log.info("feature extraction %d/%d | %.2f jobs/s | elapsed %s | ETA %s | %s %s rep=%s",
             done, total, rate, fmt_duration(elapsed), fmt_duration(eta), win.organism, win.source, win.replicate)


def fmt_duration(seconds: float) -> str:
    """Format a duration in seconds into a compact human-readable string."""
    if not math.isfinite(seconds):
        return "?"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ----------------------------- statistics ----------------------------------

def numeric_columns(rows: Sequence[Dict[str, object]]) -> List[str]:
    """Identify columns that contain numeric feature values."""
    skip = {"organism", "accession", "organism_type", "source", "replicate", "start", "end"}
    cols = []
    for k, v in rows[0].items():
        if k in skip:
            continue
        try:
            float(v)
            cols.append(k)
        except Exception:
            pass
    return cols


def summarize_by_organism(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate observed-window features into per-organism summaries.

    Currently this reports central tendency and spread for each numeric feature.
    Only observed windows are summarized here; controls are handled separately in
    :func:`hypothesis_report`."""
    obs = [r for r in rows if r["source"] == "observed"]
    cols = numeric_columns(obs)
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in obs:
        groups[str(r["organism"])].append(r)
    out = []
    for org, rs in sorted(groups.items()):
        base = {"organism": org, "organism_type": rs[0]["organism_type"], "accession": rs[0]["accession"], "n_windows": len(rs)}
        for c in cols:
            vals = np.array([safe_float(r[c]) for r in rs], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                base[c + "_mean"] = float("nan")
                base[c + "_sd"] = float("nan")
                base[c + "_ci95"] = float("nan")
            else:
                base[c + "_mean"] = float(vals.mean())
                base[c + "_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                base[c + "_ci95"] = float(1.96 * base[c + "_sd"] / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        out.append(base)
    return out


def hypothesis_report(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Compare observed windows against control windows feature by feature.

    For each organism and feature, the report stores effect sizes and empirical
    p-value style summaries. This is where the project becomes explicitly
    hypothesis-driven rather than purely descriptive."""
    cols = numeric_columns(rows)
    controls = sorted({str(r["source"]) for r in rows if r["source"] != "observed"})
    report: Dict[str, object] = {"controls": controls, "features": {}}
    for feature in cols:
        observed = np.array([safe_float(r[feature]) for r in rows if r["source"] == "observed"], dtype=float)
        observed = observed[np.isfinite(observed)]
        if len(observed) < 3:
            continue
        entry = {"observed_mean": float(observed.mean()), "observed_sd": float(observed.std(ddof=1))}
        for control in controls:
            null = np.array([safe_float(r[feature]) for r in rows if r["source"] == control], dtype=float)
            null = null[np.isfinite(null)]
            if len(null) < 3:
                continue
            diff = observed.mean() - null.mean()
            pooled = math.sqrt((observed.var(ddof=1) + null.var(ddof=1)) / 2.0) + 1e-15
            # two-sided empirical p using all null window values as exchangeable baseline
            p_emp = (np.sum(np.abs(null - null.mean()) >= abs(diff)) + 1) / (len(null) + 1)
            entry[control] = {
                "null_mean": float(null.mean()),
                "null_sd": float(null.std(ddof=1)),
                "delta": float(diff),
                "cohens_d": float(diff / pooled),
                "empirical_p_rough": float(p_emp),
            }
        report["features"][feature] = entry
    return report


def safe_float(x: object) -> float:
    """Safely coerce a value to float, returning NaN on failure."""
    try:
        return float(x)
    except Exception:
        return float("nan")


def add_fisher_centroid(rows: List[Dict[str, object]], windows: Sequence[WindowRecord], k: int = 2) -> None:
    # Across observed windows only: compute organism/window distance to observed composition centroid.
    """Post-compute Fisher–Rao distances using observed-window centroids.

    This is run after the initial feature pass because the centroid depends on the
    collection of observed windows rather than any single window in isolation."""
    vecs = []
    idxs = []
    for i, (row, win) in enumerate(zip(rows, windows)):
        if row["source"] == "observed":
            vecs.append(kmer_vector(win.sequence, k))
            idxs.append(i)
    if not vecs:
        return
    centroid = np.mean(np.vstack(vecs), axis=0)
    centroid /= centroid.sum()
    for i, vec in zip(idxs, vecs):
        rows[i]["fisher_to_observed_centroid"] = fisher_rao(vec, centroid)
    for i, row in enumerate(rows):
        if "fisher_to_observed_centroid" not in row:
            row["fisher_to_observed_centroid"] = float("nan")


def get_networkx():
    """Import and cache :mod:`networkx` lazily."""
    global nx
    if nx is None:
        try:
            import networkx as _nx
            nx = _nx
        except Exception:
            nx = False
    return None if nx is False else nx


def get_pyplot():
    """Import and cache :mod:`matplotlib.pyplot` lazily.

    Lazy imports reduce worker start-up cost and keep headless compute nodes from
    paying the plotting import penalty unless figures are actually requested."""
    global plt
    if plt is None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as _plt
            plt = _plt
        except Exception:
            plt = False
    return None if plt is False else plt


# ----------------------------- de Bruijn -----------------------------------

def debruijn_diagnostics(records: Sequence[SequenceRecord], cfg: Config) -> Dict[str, object]:
    """Run full-sequence de Bruijn graph diagnostics.

    This step is deliberately separated from the window pipeline because it is more
    about global topology than per-window statistical testing."""
    nx_mod = get_networkx()
    if nx_mod is None:
        return {"error": "networkx is not installed"}
    report: Dict[str, object] = {"k": cfg.graph_k, "per_genome": {}, "aggregate": {}}
    aggregate_edges: Counter = Counter()
    for rec in records:
        seq = rec.sequence[: min(rec.length, cfg.window * max(1, cfg.windows_per_genome))]
        edge_counts = debruijn_edge_counts(seq, cfg.graph_k)
        aggregate_edges.update(edge_counts)
        report["per_genome"][rec.name] = graph_stats_from_edges(edge_counts, cfg.graph_k)
    report["aggregate"] = graph_stats_from_edges(aggregate_edges, cfg.graph_k, communities=True)
    return report


def debruijn_edge_counts(seq: str, k: int) -> Counter:
    """Count de Bruijn graph edges implied by consecutive k-mers."""
    edges = Counter()
    for word, count in kmer_counts(seq, k).items():
        edges[(word[:-1], word[1:])] += count
    return edges


def graph_stats_from_edges(edge_counts: Counter, k: int, communities: bool = False) -> Dict[str, object]:
    """Compute graph-level summary statistics from de Bruijn edge counts."""
    nx_mod = get_networkx()
    if nx_mod is None:
        return {}
    G = nx_mod.DiGraph()
    for (u, v), w in edge_counts.items():
        G.add_edge(u, v, weight=int(w))
    possible_nodes = 4 ** (k - 1)
    possible_edges = 4 ** k
    stats: Dict[str, object] = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "node_occupancy": G.number_of_nodes() / possible_nodes,
        "edge_occupancy": G.number_of_edges() / possible_edges,
        "density": nx_mod.density(G) if G.number_of_nodes() else float("nan"),
        "scc": nx_mod.number_strongly_connected_components(G) if G.number_of_nodes() else 0,
    }
    if G.number_of_nodes():
        pr = nx_mod.pagerank(G, weight="weight")
        stats["top_pagerank"] = sorted(pr.items(), key=lambda kv: kv[1], reverse=True)[:10]
    if communities and G.number_of_nodes():
        Gu = G.to_undirected()
        comms = list(nx_mod.community.greedy_modularity_communities(Gu, weight="weight"))
        stats["num_communities_greedy"] = len(comms)
        stats["community_sizes"] = sorted([len(c) for c in comms], reverse=True)
        stats["modularity"] = nx_mod.community.modularity(Gu, comms, weight="weight")
        stats["communities"] = [sorted(list(c)) for c in comms]
    return stats


# ------------------------------- writing -----------------------------------

def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write a list of dictionaries to CSV with stable field ordering."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    extras = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames and k not in extras:
                extras.append(k)
    fieldnames += extras
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json_safe(v) for k, v in r.items()})


def write_json(path: Path, obj: object) -> None:
    """Write JSON with UTF-8 encoding and pretty indentation."""
    path.write_text(json.dumps(json_safe(obj), indent=2, sort_keys=True))


def json_safe(obj):
    """Recursively convert NumPy and non-JSON-safe objects into plain Python types."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not math.isfinite(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


# -------------------------------- plots -------------------------------------

def make_figures(summary: Sequence[Dict[str, object]], report: Dict[str, object], out: Path) -> None:
    """Generate a compact figure set from summary tables and reports."""
    plt_mod = get_pyplot()
    if plt_mod is None or not summary:
        return
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    plot_metric(summary, "gc_pct_mean", "GC%", figdir / "gc_pct.png")
    plot_metric(summary, "fractal_dim_mean", "Chaos-walk fractal dimension", figdir / "fractal_dim.png")
    plot_metric(summary, "fft_codon_z_mean", "Period-3 FFT z-score", figdir / "fft_codon_z.png")
    plot_metric(summary, "sheaf_h1_proxy_mean", "Patch coherence obstruction proxy", figdir / "sheaf_proxy.png")
    plot_effects(report, figdir / "null_effect_sizes.png")


def plot_metric(summary: Sequence[Dict[str, object]], key: str, title: str, path: Path) -> None:
    """Create a simple bar plot for one organism-level summary metric."""
    names = [str(r["organism"]) for r in summary]
    vals = [safe_float(r.get(key)) for r in summary]
    errs = [safe_float(r.get(key.replace("_mean", "_ci95"), 0.0)) for r in summary]
    colors = [TYPE_COLORS.get(str(r.get("organism_type", "unknown")), "#888888") for r in summary]
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.35), 5))
    ax.bar(names, vals, yerr=errs, color=colors, alpha=0.85, capsize=2)
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=55, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_effects(report: Dict[str, object], path: Path, top_n: int = 20) -> None:
    """Plot the strongest observed-vs-control feature effects."""
    feats = []
    for feature, entry in report.get("features", {}).items():
        ds = []
        for control, stats in entry.items():
            if isinstance(stats, dict) and "cohens_d" in stats:
                ds.append(abs(float(stats["cohens_d"])))
        if ds:
            feats.append((feature, max(ds)))
    feats = sorted(feats, key=lambda kv: kv[1], reverse=True)[:top_n]
    if not feats:
        return
    fig, ax = plt.subplots(figsize=(9, max(4, len(feats) * 0.3)))
    ax.barh([f for f, _ in feats][::-1], [d for _, d in feats][::-1])
    ax.set_title("Largest observed-vs-null effect sizes")
    ax.set_xlabel("max |Cohen's d| across controls")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Command-line interface and program entry point
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Define and parse the command-line interface.

    The CLI is intentionally explicit. Important usability note: `--full` is a real
    flag in this version and contributes its accessions cumulatively."""
    p = argparse.ArgumentParser(description="DNA Alchemy Framework v6.0")
    p.add_argument("--email", default="anonymous@example.com", help="NCBI Entrez email")
    p.add_argument("--out", default="results_v6", help="output directory")
    p.add_argument("--cache", default="genome_cache", help="FASTA cache directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--window", type=int, default=200_000)
    p.add_argument("--windows-per-genome", type=int, default=8)
    p.add_argument("--min-window", type=int, default=1_000)
    p.add_argument("--jaccard-k", type=int, default=7)
    p.add_argument("--graph-k", type=int, default=4)
    p.add_argument("--controls", default="shuffle,markov1,revcomp", help="comma list: shuffle,markov0,markov1,revcomp")
    p.add_argument("--full", action="store_true", help="include E. coli and B. subtilis; cumulative with larger tiers")
    p.add_argument("--extra", action="store_true", help="include extra bacteria/archaea/yeast chromosome; implies --full")
    p.add_argument("--ultra", action="store_true", help="include original v5/v6 large panel; implies --extra and --full")
    p.add_argument("--diverse-eukaryotes", dest="diverse_eukaryotes", action="store_true", help="add ~400 Mb eukaryote expansion: Arabidopsis, C. elegans, Drosophila, human chr22/X")
    p.add_argument("--mega", action="store_true", help="ultra + diverse-eukaryotes; the new broad taxonomy validation tier")
    p.add_argument("--fasta", action="append", default=[], help="local FASTA file; can be repeated")
    p.add_argument("--no-fetch", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--workers", type=int, default=0, help="parallel worker processes; 0 means auto = CPU cores minus one")
    p.add_argument("--chunksize", type=int, default=4, help="windows sent to each worker per batch; larger reduces overhead")
    p.add_argument("--progress-every", type=int, default=25, help="log progress every N completed windows")
    p.add_argument("--no-debruijn", action="store_true", help="skip final networkx de Bruijn graph diagnostics")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    """Translate parsed CLI arguments into a frozen runtime configuration."""
    return Config(
        out=Path(args.out), cache=Path(args.cache), email=args.email, seed=args.seed,
        window=args.window, windows_per_genome=args.windows_per_genome, min_window=args.min_window,
        jaccard_k=args.jaccard_k, graph_k=args.graph_k,
        controls=[c.strip() for c in args.controls.split(",") if c.strip()],
        make_plots=not args.no_plots, fetch=not args.no_fetch,
        workers=args.workers, chunksize=args.chunksize, progress_every=args.progress_every,
        debruijn=not args.no_debruijn,
    )


def load_records(args: argparse.Namespace, cfg: Config) -> List[SequenceRecord]:
    """Load all input sequences from NCBI and local FASTA files.

    The function also normalizes organism types after loading so that the rest of
    the pipeline can rely on a consistent taxonomy vocabulary."""
    records: List[SequenceRecord] = []
    for name, acc in accession_set(args).items():
        rec = fetch_ncbi(name, acc, cfg)
        if rec and rec.length:
            records.append(rec)
    for fp in args.fasta:
        rec = read_fasta(Path(fp))
        if rec.length:
            records.append(rec)
    fixed = []
    for r in records:
        typ = infer_organism_type(r.name, r.accession, r.organism_type)
        fixed.append(SequenceRecord(r.name, r.accession, typ, r.sequence))
    unknowns = [r.name for r in fixed if r.organism_type == "unknown"]
    if unknowns:
        log.warning("%d records have unknown taxonomy, not defaulting them to bacterium: %s", len(unknowns), ", ".join(unknowns[:8]))
    return fixed


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for command-line execution."""
    args = parse_args(argv)
    cfg = build_config(args)
    setup_logging(cfg.out)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    log.info("DNA Alchemy Framework v6.0")
    log.info("config: %s", asdict(cfg))
    records = load_records(args, cfg)
    if not records:
        log.error("no sequences loaded")
        return 2
    log.info("loaded %d sequences; total %.2f Mb", len(records), sum(r.length for r in records) / 1e6)

    manifest = [{"name": r.name, "accession": r.accession, "organism_type": r.organism_type,
                 "length_bp": r.length, "sha256": r.sha256} for r in records]
    write_json(cfg.out / "manifest.json", {"config": asdict(cfg), "sequences": manifest})

    windows = list(iter_windows(records, cfg))
    log.info("created %d windows including controls", len(windows))

    rows = compute_all_features(windows, cfg)
    add_fisher_centroid(rows, windows, k=2)

    write_csv(cfg.out / "summary_windows.csv", rows)
    summary = summarize_by_organism(rows)
    write_csv(cfg.out / "organism_summary.csv", summary)
    report = hypothesis_report(rows)
    write_json(cfg.out / "hypothesis_report.json", report)

    if cfg.debruijn:
        log.info("building de Bruijn diagnostics")
        db_report = debruijn_diagnostics(records, cfg)
        write_json(cfg.out / "debruijn_report.json", db_report)
    else:
        log.info("skipping de Bruijn diagnostics (--no-debruijn)")

    if cfg.make_plots:
        make_figures(summary, report, cfg.out)

    log.info("done: %s", cfg.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
