# Machine Learning-Based Structural Damage Classification and Sensor Optimization in Truss Systems

An end-to-end unsupervised and supervised machine learning framework for dynamic structural health monitoring (SHM), modal parameter analysis (MAC), and sensor layout optimization under measurement noise.

---

## Project Overview

This repository provides open-source implementation codes and datasets for structural damage identification, modal parameter evaluation, and sensor robustness benchmarking on truss structures. The framework combines:

- **Unsupervised Learning (K-Means Clustering):** Automated grouping and severity classification of modal assurance criterion (MAC) features across damage scenarios.
- **Supervised Learning (Random Forest Classification):** Multi-class structural damage pattern recognition.
- **Feature Explainability:** Modal parameter importance evaluation across dynamic modes and structural degrees of freedom (DOFs).
- **Sensor Layout Robustness:** Performance assessment of sensor reduction (8-DOF, 4-DOF, 2-DOF) subjected to varying operational noise levels (0% to 15% Gaussian noise).

---

## Repository Structure

- `figures/` : Visualizations of clusters, confusion matrices, and noise robustness.
- `01_kmeans_clustering.py` : Unsupervised clustering analysis on MAC features.
- `02_damage_heatmap.py` : Spatial damage localization and visualization.
- `03_rf_classification.py` : Supervised Random Forest multi-class damage identification.
- `04_feature_importance.py` : Permutation & Gini feature importance evaluation.
- `05_sensor_robustness.py` : Benchmark of 8-DOF, 4-DOF, and 2-DOF under 0-15% noise.
- `requirements.txt` : Python dependencies for the pipeline.

---

## Installation & Usage

Ensure Python 3.8+ is installed. Install dependencies:

```bash
pip install -r requirements.txt
```

Execute the pipeline sequentially:

```bash
python 01_kmeans_clustering.py
python 02_damage_heatmap.py
python 03_rf_classification.py
python 04_feature_importance.py
python 05_sensor_robustness.py
```

---

## Methodology Summary

- **Modal Assurance Criterion (MAC):** Quantifies modal vector correlation between healthy and damaged states.
- **K-Means Clustering:** Identifies structural damage clusters without requiring prior manual thresholding.
- **Random Forest Classifier:** Employs ensemble decision trees to handle nonlinear multi-dimensional modal interactions.
- **Sensor Optimization:** Compares diagnostic accuracy across 8, 4, and 2 sensor configurations across realistic operational signal-to-noise ratios.
