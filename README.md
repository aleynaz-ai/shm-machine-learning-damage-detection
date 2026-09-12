Machine Learning-Based Structural Damage Classification and Sensor Optimization in Truss Systems
An end-to-end unsupervised and supervised machine learning framework for dynamic structural health monitoring (SHM), modal parameter analysis (MAC), and sensor layout optimization under measurement noise.

Project Overview
This repository provides open-source implementation codes and datasets for structural damage identification, modal parameter evaluation, and sensor robustness benchmarking on truss structures. The framework combines:

Unsupervised Learning (K-Means Clustering): Automated grouping and severity classification of modal assurance criterion (MAC) features across damage scenarios.

Supervised Learning (Random Forest Classification): Multi-class structural damage pattern recognition.

Feature Explainability: Modal parameter importance evaluation across dynamic modes and structural degrees of freedom (DOFs).

Sensor Layout Robustness: Performance assessment of sensor reduction (8-DOF, 4-DOF, 2-DOF) subjected to varying operational noise levels (0% to 15% Gaussian noise).

Associated Publication & Citation
Note: This repository accompanies forthcoming research on machine learning-driven damage classification and sensor optimization in truss structures.

Full citation, publication metadata, and DOI links will be updated here upon formal publication.

If you utilize this dataset, codes, or methodology in your research, please attribute the work accordingly:
Machine Learning-Driven Damage Classification and Sensor Placement Optimization in Truss Structures (Forthcoming Book Chapter).

Repository Structure
figures/

Damage_Heatmap.png

Feature_Importance_Plot.png

kmeans_clusters.png

kmeans_validation.png

RF_Classification_Results.png

Sensor_Robustness_Comparative.png

01_kmeans_clustering.py

02_damage_heatmap.py

03_rf_classification.py

04_feature_importance.py

05_sensor_robustness.py

Clustered_Results_new.xlsx

MAC_2DOF.xlsx

MAC_4DOF.xlsx

MAC_8DOF.xlsx

MAC_Results.xlsx

requirements.txt

README.md

Installation & Requirements
Ensure you have Python 3.8+ installed. Install the required dependencies using:

pip install -r requirements.txt

Pipeline Workflow
Run the modules sequentially to replicate the complete analysis:

Unsupervised Damage Clustering:
python 01_kmeans_clustering.py

Damage Distribution Heatmap:
python 02_damage_heatmap.py

Random Forest Classification:
python 03_rf_classification.py

Modal Feature Importance:
python 04_feature_importance.py

Sensor Robustness & Noise Evaluation:
python 05_sensor_robustness.py

Methodology Summary
Modal Assurance Criterion (MAC): Quantifies modal vector correlation between healthy and damaged states.

K-Means Clustering: Identifies structural damage clusters without requiring prior manual thresholding.

Random Forest Classifier: Employs ensemble decision trees to handle nonlinear multi-dimensional modal interactions.

Sensor Optimization: Compares diagnostic accuracy across 8, 4, and 2 sensor configurations across realistic operational signal-to-noise ratios.
