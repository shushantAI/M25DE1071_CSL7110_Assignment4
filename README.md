# CSL7110 - Assignment 4: Clustering and PageRank

**Name:** Shushant Kumar Tiwari
**Roll No:** M25DE1071
**Assignment:** Assignment 4 — Clustering and PageRank

---

## Overview

This repository contains my full solution for Assignment 4. The assignment has
three independent parts:

| Part | Topic | Dataset |
| --- | --- | --- |
| 1 | k-center (Farthest-First Traversal) and k-means++ | UCI Spambase (`spambase.data`, 4601 × 58) |
| 2 | Inverted index / simple search engine | 7 webpages in `data/q2/webpages/` + `actions.txt` |
| 3 | PageRank on Spark (40 iters, β = 0.8) | `small.txt`, `whole.txt` from the course's PySpark-PageRank repo |

All code is written in Python. Part 1 uses NumPy, Part 2 uses only the
standard library, and Part 3 uses PySpark.

---

## Folder structure

```
M25DE1071_CSL7110_Assignment4/
├── README.md
├── requirements.txt
├── data/
│   ├── q1/
│   │   ├── spambase.data
│   │   └── spambase.names
│   ├── q2/
│   │   ├── actions.txt
│   │   ├── answers.txt
│   │   └── webpages/
│   │       ├── references
│   │       ├── stack_cprogramming
│   │       ├── stack_datastructure_wiki
│   │       ├── stack_oracle
│   │       ├── stacklighting
│   │       ├── stackmagazine
│   │       └── stackoverflow
│   └── q3/
│       ├── small.txt      (download from the GitHub link below)
│       └── whole.txt      (download from the GitHub link below)
├── src/
│   ├── part1_clustering.py
│   ├── part2_search.py
│   └── part3_pagerank.py
└── notebooks/
    └── Assignment4_M25DE1071.ipynb   

```

---

## Setup

I tested everything with Python 3.11 and Java 21. Any Java ≥ 8 works for
PySpark.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For Part 3 you also need to grab the graph files once:

```bash
mkdir -p data/q3
curl -L -o data/q3/small.txt  https://raw.githubusercontent.com/pnijhara/PySpark-PageRank/main/graph/small.txt
curl -L -o data/q3/whole.txt  https://raw.githubusercontent.com/pnijhara/PySpark-PageRank/main/graph/whole.txt
```

---

## How to run

### Option A — run the three scripts

```bash
# Part 1
python -m src.part1_clustering --path data/q1/spambase.data --k 10 --k1 50

# Part 2
python -m src.part2_search --pages data/q2/webpages \
                           --actions data/q2/actions.txt \
                           --answers data/q2/answers.txt

# Part 3
python -m src.part3_pagerank --edges data/q3/small.txt --iters 40 --beta 0.8
python -m src.part3_pagerank --edges data/q3/whole.txt --iters 40 --beta 0.8

# Part 3 (optional) - cross-check the PySpark implementation against a
# NumPy reference on a 3-cycle and a small skewed graph.
python -m src.part3_sanity
```

### Option B — run the consolidated notebook

```bash
jupyter notebook notebooks/Assignment4_M25DE1071.ipynb
```

The notebook has one section per part with short explanation cells,
runnable code cells, and output cells.

