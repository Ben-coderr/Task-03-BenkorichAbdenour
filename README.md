# 🎯 Data Science Project 3: Customer Segmentation
**DecodeLabs Internship | Batch 2026**

## 📖 Overview
An Unsupervised Learning project leveraging K-Means Clustering and Principal Component Analysis (PCA) to discover actionable business personas within a high-dimensional retail dataset.

## 🏗 Architecture & Features
* **Dimensionality Compression:** Uses PCA to map 20+ raw behavioral/financial features down to a mathematically dense 2D/3D subspace, maintaining an enterprise 95% explained variance threshold.
* **Diagnostic Gatekeepers:** Employs dual mathematical proofs to dynamically select the optimal number of clusters ($K$):
  1. The **Elbow Method** (tracking Within-Cluster Sum of Squares).
  2. The **Silhouette Score** (measuring inter-cluster distance vs. intra-cluster cohesion).
* **Strategic Persona Matrix:** Maps raw mathematical cluster centroids back to their original human-centric feature scales via an inverse transform $(C_{orig} = C_{scaled} \odot \sigma + \mu)$.
* **Business Translation:** Translates abstract coordinates into specific strategic groups (e.g., *High-Value Trendsetters*, *Budget-Conscious Explorers*).

## 💻 Repository Structure
* `project3_solution.ipynb`: Professional Jupyter Notebook encompassing the scaling, PCA projection, clustering algorithms, and 3D visual plotting.
* `project3_solution.py`: The executable Python pipeline.
* `customer_segmentation_results.csv`: The dataset appended with dynamically assigned strategic segments.
* `plots/`: Contains visual proofs, including PCA scatter projections, spending heatmaps, and Silhouette score validations.

## 🚀 Getting Started
```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn seaborn matplotlib jupyter

# 2. Run the Notebook
jupyter notebook project3_solution.ipynb

# 3. Or run the CLI script
python project3_solution.py
```

## 📊 Impact
Raw behavioral noise is mathematically synthesized into distinct, actionable cohorts, allowing marketing systems to execute personalized, targeted campaigns.

---
*Authored with strict adherence to DecodeLabs architectural guidelines.*
