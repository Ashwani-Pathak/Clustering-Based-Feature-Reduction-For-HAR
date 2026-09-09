# Optimizing ML Execution: Dimensionality Reduction via Feature Clustering

## 📌 Strategic Overview
High-dimensional datasets often introduce noise, increase computational overhead, and lead to suboptimal model performance due to the curse of dimensionality. This project explores an unconventional yet highly effective approach to feature selection: **treating features as data points and clustering them**.

By applying this methodology to the **Human Activity Recognition (HAR) Using Smartphones dataset**, we successfully isolated the most informative signals from 561 raw sensor readings. The result is a drastically optimized workflow: 
*   **12x reduction** in wall-clock training time.
*   **~8.6% absolute improvement** in classification accuracy.
*   **91% reduction** in feature space complexity.

This repository contains the exploratory data analysis, data processing pipelines, and the final optimized Gaussian Naive Bayes classification model.

---

## 🛠 Tech & Tools
*   **Languages:** Python
*   **Libraries:** Scikit-Learn (KMeans, GaussianNB, Pipeline, StandardScaler, LabelEncoder), Pandas, NumPy, Matplotlib

---

## 🧠 Methodology: Inverting the Data Matrix
Standard clustering groups similar data points. In this project, we pivot the perspective:
1.  **Normalization:** All numerical sensor features are scaled using `StandardScaler` to ensure uniform ranges.
2.  **Transposition:** We transpose the dataset matrix. Data points become the dimensions, and the 561 original features become the new "data points."
3.  **Clustering (K-Means):** We group these 561 features into `K` clusters. Because similar features will have similar distributions across the original dataset, they fall into the same cluster.
4.  **Feature Selection:** We randomly select one representative feature from each cluster, effectively removing redundant sensor readings.
5.  **Classification:** A Gaussian Naive Bayes model is trained on this aggressively reduced feature set.

---

## 📊 Results & Metrics Tracking

We established a baseline using all 561 features and compared it against our K-Means optimized pipeline (using 50 clusters).

| Metric | Baseline (All Features) | Optimized (K-Means, n=50) | Impact / Delta |
| :--- | :--- | :--- | :--- |
| **Feature Count** | 561 | 50 | **-91% (511 features removed)** |
| **Accuracy** | 73.15% | 81.78% | **+8.63% Absolute Gain** |
| **Training Time** | 0.1695 sec | 0.0139 sec | **91.8% Faster (12.2x speedup)** |

### Why did accuracy improve?
Naive Bayes assumes feature independence. The raw dataset contained highly correlated and redundant sensor readings, which violates this assumption and degrades the model's performance. By clustering and selecting representatives, we forced feature independence and removed noise, significantly improving the model's predictive power.

---

## 📈 Performance Visualizations

To determine the optimal execution roadmap, we experimented with different cluster sizes (`n_clusters` = 40, 50, 60) to observe the trade-off between dimensionality, accuracy, and computational load.
(download.png)
### 1. Accuracy vs. Feature Dimensionality


*Observation:* Peak accuracy is achieved at approximately 50 clusters. Reducing features too aggressively (e.g., 40) strips away vital information, while increasing them (e.g., 60) begins to reintroduce correlated noise, slightly dampening the Naive Bayes performance.

### 2. Training Time Optimization

*Observation:* Wall-clock time scales linearly with the number of retained features. Even at 60 features, the training time is an order of magnitude faster than the 561-feature baseline, proving the viability of this strategy for rapid model iteration.

---

## 🚀 How to Run the Code

1.  Clone the repository and navigate to the project directory.
2.  Ensure you have the required dependencies installed:
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```
3.  Execute the main script to run the pipelines and output the metrics:
    ```bash
    python main.py
    ```

## 💡 Key Takeaway
Dimensionality reduction isn't strictly limited to PCA or target-aware feature selection. Unsupervised clustering on a transposed data matrix is a powerful, intuitive tool to distill high-dimensional datasets into lean, highly performant feature sets.
