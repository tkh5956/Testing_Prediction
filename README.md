# Beyond Point Estimates: Reliable Evaluation of Prediction Performance Metrics under Clustered Data

This repository provides the R implementation for the paper:
**"Beyond Point Estimates: Reliable Evaluation of Prediction Performance Metrics under Clustered Data"**.

The repository contains simulation and application code for statistical inference on prediction performance metrics under clustered data. The methods express classification performance metrics as smooth functionals of confusion-matrix probabilities and use cluster-adjusted sandwich variance estimation with delta-method inference.

## 🚀 Key Features

* **Cluster-Adjusted Inference:** Provides confidence intervals and hypothesis tests for prediction performance metrics while accounting for within-cluster dependence.
* **Unified Metric Framework:** Handles a broad class of metrics expressible as smooth functions of confusion-matrix probabilities, including sensitivity, specificity, accuracy, F1 score, macro-F1, micro-F1, and Matthews correlation coefficient.
* **Single-Model Evaluation:** Supports superiority testing against a prespecified performance benchmark.
* **Paired Model Comparison:** Supports noninferiority testing and paired comparisons between candidate and reference models evaluated on the same clustered dataset.
* **Sandwich Variance Estimation:** Uses cluster-level sandwich variance estimators to propagate dependence into uncertainty estimates without explicitly modeling the within-cluster correlation structure.
* **Power and Sample Size Approximation:** Provides pilot-based power and sample size calculations for prospective performance evaluation studies.

## 📂 Folder Overview

* `/simulation`: Code for reproducing the main simulation results and manuscript tables.
* `/application`: Code for the Human Activity Recognition data application.

## Reference
If you use this code, please cite the following article:

Hong, T., Lim, D. & Bae, W. (2026). *Beyond Point Estimates: Reliable Evaluation of Prediction Performance Metrics under Clustered Data*. arXiv.
