# Hierarchical Clustering (Agglomerative) — Manim Visualization

This project contains a **Manim** animation (3Blue1Brown style) that demonstrates **hierarchical agglomerative clustering** and shows how a **dendrogram** is created step-by-step.

## Dataset
We use a tiny, classroom-friendly subset of the **UCI Iris dataset** (8 flowers) with two features:
- **petal length (cm)**
- **petal width (cm)**

The Iris dataset is hosted by the UCI Machine Learning Repository. See: https://archive.ics.uci.edu/ml/datasets/Iris

The file used here is: `data/iris_8.csv`.

## What the animation shows
1. Scatterplot of the 8 points (petal length vs petal width)
2. At each step:
   - compute the closest pair of clusters (using *average linkage*)
   - highlight the merged clusters in the scatterplot
   - add the corresponding branch to the dendrogram at the merge distance (“height”)
3. Final dendrogram shows the complete hierarchy.

## Quickstart (recommended)

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Render the video
Low quality preview:
```bash
manim -pql hierarchical_clustering.py HierarchicalClusteringDemo
```

Higher quality (still reasonably fast):
```bash
manim -pqh hierarchical_clustering.py HierarchicalClusteringDemo
```

Manim will place outputs under `media/videos/...`.

## One-command render
On macOS/Linux:
```bash
bash render.sh
```

## Notes
- Manim requires **ffmpeg** and some system libraries; if installation fails, consult the Manim Community installation guide.
- This project is offline-friendly: the dataset CSV is included locally.

