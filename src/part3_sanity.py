"""Sanity checks for the PySpark PageRank implementation.

Two cases executed under one SparkContext:
  (a) 3-node directed cycle -> uniform stationary distribution 1/3.
  (b) a small skewed graph -> compared against a numpy power iteration.

If both agree within 1e-4, the Spark implementation is behaviourally
correct on the classroom datasets (small.txt, whole.txt) too.
"""
import os
import tempfile
import numpy as np

from src.part3_pagerank import pagerank, parse_edges
from pyspark import SparkConf, SparkContext


def _numpy_pagerank(edges, n, iters=40, beta=0.8):
    out_deg = np.zeros(n)
    for u, v in edges:
        out_deg[u] += 1.0
    M = np.zeros((n, n))
    for u, v in edges:
        M[v, u] = 1.0 / out_deg[u]
    r = np.ones(n) / n
    teleport = (1 - beta) / n * np.ones(n)
    for _ in range(iters):
        dangling = r * (out_deg == 0).astype(float)
        r = teleport + beta * (M @ r) + beta * dangling.sum() / n
    return r


def _write_edges(tmpdir, name, edges):
    p = os.path.join(tmpdir, name)
    with open(p, "w") as f:
        for u, v in edges:
            f.write(f"{u} {v}\n")
    return p


def main():
    conf = (
        SparkConf()
        .setAppName("pr-sanity")
        .setMaster("local[1]")
        .set("spark.ui.showConsoleProgress", "false")
        .set("spark.default.parallelism", "1")
        .set("spark.sql.shuffle.partitions", "1")
    )
    sc = SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel("ERROR")

    with tempfile.TemporaryDirectory() as td:
        cycle = [(0, 1), (1, 2), (2, 0), (2, 0)]
        p = _write_edges(td, "cycle.txt", cycle)
        got, n = pagerank(parse_edges(sc, p), 25, 0.8)
        print("3-cycle ranks:", {k: round(v, 4) for k, v in sorted(got.items())})
        for k in got:
            assert abs(got[k] - 1 / 3) < 1e-4, f"3-cycle failed at node {k}"
        print("[OK] 3-cycle -> uniform 1/3")

        skewed = [
            (0, 1), (0, 2), (1, 2), (2, 0),
            (3, 2), (3, 4), (4, 2)
        ]
        p = _write_edges(td, "skewed.txt", skewed)
        got, n = pagerank(parse_edges(sc, p), 25, 0.8)
        ref = _numpy_pagerank(skewed, n, 25, 0.8)
        print("skewed ranks (spark):", {k: round(got[k], 6) for k in sorted(got)})
        print("skewed ranks (numpy):", {k: round(ref[k], 6) for k in range(n)})
        max_err = max(abs(got[k] - ref[k]) for k in range(n))
        print(f"max |spark - numpy| = {max_err:.2e}")
        assert max_err < 1e-4, "skewed graph mismatch"
        print("[OK] skewed graph matches numpy power iteration")

    sc.stop()


if __name__ == "__main__":
    main()
