"""Part 3 - PageRank using PySpark."""

from __future__ import annotations
import argparse
import os
from typing import List, Tuple

from pyspark import SparkConf, SparkContext


def parse_edges(sc: SparkContext, path: str):
    """Read edge list and return deduplicated (src, dst) RDD."""
    raw = sc.textFile(path)
    return (
        raw.map(lambda ln: ln.strip())
           .filter(lambda ln: ln and not ln.startswith("#"))
           .map(lambda ln: ln.split())
           .filter(lambda xs: len(xs) >= 2)
           .map(lambda xs: (int(xs[0]), int(xs[1])))
           .distinct()
    )


def pagerank(edges_rdd, num_iters: int = 40, beta: float = 0.8):
    """Power-iteration PageRank."""
    sc = edges_rdd.context

    # build adjacency list and cache
    links = (
        edges_rdd.groupByKey()
                 .mapValues(list)
                 .cache()
    )
    links.count()

    # all vertices in the graph
    vertices = sorted(
        edges_rdd.flatMap(lambda e: [e[0], e[1]])
                 .distinct()
                 .collect()
    )
    n = len(vertices)
    source_set = set(links.keys().collect())

    ranks = {v: 1.0 / n for v in vertices}
    teleport = (1.0 - beta) / n

    for _ in range(num_iters):
        # contributions from each edge
        ranks_rdd = sc.parallelize(list(ranks.items()), numSlices=4)
        contribs = (
            links.join(ranks_rdd)
                 .flatMap(
                     lambda kv: [
                         (d, kv[1][1] / len(kv[1][0])) for d in kv[1][0]
                     ]
                 )
                 .reduceByKey(lambda a, b: a + b)
                 .collectAsMap()
        )

        dangling_rank = sum(r for v, r in ranks.items() if v not in source_set)
        dangling_share = beta * dangling_rank / n

        ranks = {
            v: teleport + beta * contribs.get(v, 0.0) + dangling_share
            for v in ranks
        }

    return ranks, n


def _format_rank_list(title: str, items: List[Tuple[int, float]]) -> str:
    out = [title]
    for nid, r in items:
        out.append(f"  node {nid:>6d}   rank = {r:.8f}")
    return "\n".join(out)


def run(edges_path: str, iters: int, beta: float) -> None:
    conf = (
        SparkConf()
        .setAppName("pagerank-a4")
        .setMaster("local[2]")
        .set("spark.driver.memory", "1g")
        .set("spark.ui.showConsoleProgress", "false")
        .set("spark.ui.enabled", "false")
    )
    sc = SparkContext.getOrCreate(conf=conf)
    sc.setLogLevel("ERROR")

    print("=" * 60)
    print(f"Running PageRank on: {edges_path}")
    print(f"iterations = {iters}, beta = {beta}")
    print("=" * 60)

    edges = parse_edges(sc, edges_path).cache()
    m_unique = edges.count()

    ranks, n = pagerank(edges, iters, beta)
    print(f"Nodes n = {n}, unique directed edges after dedup = {m_unique}")

    items = sorted(ranks.items(), key=lambda kv: kv[1], reverse=True)
    top5 = [(int(k), v) for k, v in items[:5]]
    bot5 = [(int(k), v) for k, v in sorted(ranks.items(), key=lambda kv: kv[1])[:5]]

    print(_format_rank_list("Top 5 nodes by PageRank:", top5))
    print(_format_rank_list("Bottom 5 nodes by PageRank:", bot5))

    total = sum(ranks.values())
    print(f"Sanity: sum(r) = {total:.6f} (should be ~1.0)")

    if os.path.basename(edges_path).lower().startswith("small"):
        ok = abs(top5[0][1] - 0.036) < 0.005
        print(
            f"Check on small graph: expected top score ~= 0.036, "
            f"got {top5[0][1]:.4f} -- {'OK' if ok else 'CHECK'}"
        )

    sc.stop()


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default="data/q3/small.txt")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--beta", type=float, default=0.8)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(a.edges, a.iters, a.beta)
