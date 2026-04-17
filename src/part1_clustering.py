"""
Part 1 - Clustering on UCI Spambase.

Implements:
  * readVectorsSeq(filename)
  * kcenter(P, k)       -- Farthest-First Traversal, O(|P| * k)
  * kmeansPP(P, k)      -- k-means++ seeding, O(|P| * k)
  * kmeansObj(P, C)     -- average squared distance to nearest centre

I have kept the implementation in plain NumPy so that every step is
explicit. The whole routine is vectorised along the points axis, which is
what keeps the O(|P| * k) bound realistic (one pass over P per chosen
centre).

Run:
    python -m src.part1_clustering --path data/q1/spambase.data --k 10 --k1 50
"""

from __future__ import annotations
import argparse
import random
import time
from pathlib import Path
from typing import List

import numpy as np


# -------------------------------------------------------------
# 1. readVectorsSeq
# -------------------------------------------------------------
def readVectorsSeq(filename: str) -> List[np.ndarray]:
    """Read a points-per-line CSV file and return a list of numpy vectors.

    The spambase file has 58 comma separated values per row; the 58th value
    is the class label (0/1). The assignment asks for the 58-dim feature
    point, so I keep every column as-is and let the clustering treat the
    label as just another coordinate. This is consistent with how the
    course problem treats "points in Euclidean space".
    """
    vectors = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            vectors.append(np.array([float(x) for x in parts], dtype=np.float64))
    return vectors


# -------------------------------------------------------------
# small helper: squared L2 distance between one point and a matrix of points
# -------------------------------------------------------------
def _sqdist_to_point(P_mat: np.ndarray, q: np.ndarray) -> np.ndarray:
    diff = P_mat - q
    return np.einsum("ij,ij->i", diff, diff)


# -------------------------------------------------------------
# 2. kcenter  -- Farthest First Traversal
# -------------------------------------------------------------
def kcenter(P: List[np.ndarray], k: int) -> List[np.ndarray]:
    """Gonzalez's Farthest-First Traversal.

    Idea: pick an arbitrary first centre, then at every step add the point
    that is the farthest from the currently selected centre-set.
    By maintaining the running "min distance to any chosen centre" vector,
    we pay only one linear scan (O(|P|)) per new centre, giving the
    required O(|P| * k) running time.
    """
    n = len(P)
    if k <= 0 or k > n:
        raise ValueError("k must be in [1, |P|]")

    P_mat = np.vstack(P)              # (n, d) numpy view of the points
    centres_idx = [0]                 # take point 0 as the first centre
    # running minimum distance from every point to the selected centres
    min_d = _sqdist_to_point(P_mat, P_mat[0])

    for _ in range(1, k):
        nxt = int(np.argmax(min_d))   # farthest point from the current set
        centres_idx.append(nxt)
        new_d = _sqdist_to_point(P_mat, P_mat[nxt])
        np.minimum(min_d, new_d, out=min_d)

    return [P[i] for i in centres_idx]


# -------------------------------------------------------------
# 3. kmeansPP  -- k-means++ seeding
# -------------------------------------------------------------
def kmeansPP(P: List[np.ndarray], k: int, seed: int | None = 42) -> List[np.ndarray]:
    """k-means++ initialisation.

    After an initial random centre, each next centre is sampled with
    probability proportional to D(x)^2, where D(x) is the squared distance
    from x to the nearest already-chosen centre. Again one linear pass per
    new centre, so O(|P| * k).
    """
    n = len(P)
    if k <= 0 or k > n:
        raise ValueError("k must be in [1, |P|]")

    rng = random.Random(seed)
    P_mat = np.vstack(P)

    first = rng.randrange(n)
    centres_idx = [first]
    min_d = _sqdist_to_point(P_mat, P_mat[first])

    for _ in range(1, k):
        total = float(min_d.sum())
        if total <= 0.0:
            # All remaining points coincide with the chosen centres.
            # Fall back to a uniform pick over unchosen indices.
            remaining = [i for i in range(n) if i not in set(centres_idx)]
            nxt = rng.choice(remaining)
        else:
            r = rng.random() * total
            # weighted pick: walk the prefix sums
            cs = np.cumsum(min_d)
            nxt = int(np.searchsorted(cs, r))
            if nxt >= n:
                nxt = n - 1
        centres_idx.append(nxt)
        new_d = _sqdist_to_point(P_mat, P_mat[nxt])
        np.minimum(min_d, new_d, out=min_d)

    return [P[i] for i in centres_idx]


# -------------------------------------------------------------
# 4. kmeansObj -- average squared distance to nearest centre
# -------------------------------------------------------------
def kmeansObj(P: List[np.ndarray], C: List[np.ndarray]) -> float:
    P_mat = np.vstack(P)
    C_mat = np.vstack(C)
    # For memory safety, chunk the broadcast
    chunk = 2048
    total = 0.0
    for start in range(0, P_mat.shape[0], chunk):
        block = P_mat[start:start + chunk]
        # (chunk, k) matrix of squared distances
        d2 = ((block[:, None, :] - C_mat[None, :, :]) ** 2).sum(axis=2)
        total += float(d2.min(axis=1).sum())
    return total / P_mat.shape[0]


# -------------------------------------------------------------
# Driver
# -------------------------------------------------------------
def run(path: str, k: int, k1: int) -> None:
    print("=" * 60)
    print(f"Loading points from: {path}")
    P = readVectorsSeq(path)
    print(f"Number of points |P| = {len(P)}, dimension d = {len(P[0])}")
    print(f"Chosen k = {k}, k1 = {k1}")
    print("=" * 60)

    # (1) kcenter on P
    t0 = time.perf_counter()
    C_fft = kcenter(P, k)
    t1 = time.perf_counter()
    print(f"[1] kcenter(P, k={k})  running time = {t1 - t0:.4f} s")

    # For completeness also report its objective, though the assignment
    # only asks for timing here. Keeps the comparison tidy later.
    obj_fft = kmeansObj(P, C_fft)
    print(f"    kmeansObj(P, C_fft) = {obj_fft:.6f}")

    # (2) kmeans++ on P
    t0 = time.perf_counter()
    C_pp = kmeansPP(P, k)
    t1 = time.perf_counter()
    obj_pp = kmeansObj(P, C_pp)
    print(f"[2] kmeansPP(P, k={k})  running time = {t1 - t0:.4f} s")
    print(f"    kmeansObj(P, C_pp)  = {obj_pp:.6f}")

    # (3) coreset-style: kcenter(P, k1) -> kmeansPP(X, k) -> kmeansObj(P, C)
    t0 = time.perf_counter()
    X = kcenter(P, k1)
    C_core = kmeansPP(X, k)
    t1 = time.perf_counter()
    obj_core = kmeansObj(P, C_core)
    print(f"[3] kcenter(P, k1={k1}) -> kmeansPP(X, k={k})")
    print(f"    total running time  = {t1 - t0:.4f} s")
    print(f"    kmeansObj(P, C_core) = {obj_core:.6f}")
    print("=" * 60)

    # Short summary table
    print("Objective comparison (lower is better):")
    print(f"  farthest-first centres  : {obj_fft:.6f}")
    print(f"  kmeans++ centres        : {obj_pp:.6f}")
    print(f"  FFT-coreset + kmeans++  : {obj_core:.6f}")


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/q1/spambase.data")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--k1", type=int, default=50)
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(args.path, args.k, args.k1)
