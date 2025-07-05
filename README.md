# Deep Learning for Image Clustering and Bias Analysis

**Course:** Data Analytics for Business (2024/2025)  
**Authors:** Matteo Mastroianni, Matteo Cantafio, Alessia Froio, Giuseppe Rizzo


*t-SNE visualization of clusters based on semantically-rich features, achieving a Silhouette Score of 0.72.*

## Project Overview

This university group project provides an in-depth analysis of the **CelebA (CelebFaces Attributes) dataset**. The study moves from a baseline clustering approach using generic embeddings to a more sophisticated methodology focused on semantically-rich features, inspired by the work of Anzalone et al.

The primary goals are:
1.  To compare different feature representations (FaceNet embeddings vs. semantic attributes) for clustering facial images using K-Means.
2.  To develop and evaluate predictive models for the subjective "Attractive" attribute, with a strong focus on fairness and performance disparities across different clusters.
3.  To leverage model interpretability techniques (**SHAP**) to investigate feature importance and detect potential biases in the models' decision-making process.

---

## Key Features & Methodology

The project is structured in several key phases, all implemented in a Python Jupyter Notebook:

*   **Data Preprocessing:** Cleaning and preparing the CelebA attribute file (`.csv`) and managing the large-scale image dataset (~200,000 images) using **Pandas**, **NumPy**, and **OpenCV**.
*   **Baseline Approach:**
    *   **Feature Engineering:** Generated 128-dimension embeddings using a pre-trained **FaceNet** model (**TensorFlow/Keras**).
    *   **Dimensionality Reduction:** Applied **PCA** (**Scikit-learn**) to reduce noise and identify key variance components.
    *   **Clustering:** Performed K-Means clustering and evaluated cluster separation using the Silhouette Score and t-SNE visualizations (**Matplotlib**, **Seaborn**).
*   **Advanced Approach (Anzalone et al. method):**
    *   **Semantic Feature Extraction:** Utilized a modified, pre-trained **MobileNetV2** to predict a 37-dimensional binary vector of facial attributes for each image.
    *   **Improved Clustering:** Applied K-Means on these semantic features, achieving a significantly higher **Silhouette Score of 0.72**.
*   **Predictive Modeling & Fairness Analysis:**
    *   Trained `RandomForest` and `LogisticRegression` models (**Scikit-learn**) to predict the "Attractive" attribute.
    *   Investigated the impact of feature selection and data balancing techniques (cluster-based undersampling).
    *   Conducted a disaggregated performance evaluation across clusters to identify performance disparities.
*   **Model Interpretability:**
    *   Applied **SHAP (SHapley Additive exPlanations)** to explain model predictions, revealing how feature importance varies across different demographic clusters and highlighting potential sources of bias.

