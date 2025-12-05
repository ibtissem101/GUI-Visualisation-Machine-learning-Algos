#  Machine Learning Visualization GUI

A modern, interactive desktop application for visualizing **Unsupervised Learning** algorithms. Built with Python and CustomTkinter for a sleek, professional interface.

>  **Note:** This application currently supports **Unsupervised Learning algorithms only**. Supervised learning features may be added in future updates.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![CustomTkinter](https://img.shields.io/badge/CustomTkinter-5.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

##  Features

###  Data Loading & Preprocessing
- Load datasets in **CSV**, **JSON**, and **XLSX** formats
- Automatic detection of missing values (including placeholders like `?`, `N/A`, `NULL`, etc.)
- **Per-column preprocessing options:**
  - Missing value handling (Mean, Median, Mode, Forward/Backward Fill)
  - Scaling (Min-Max, Z-Score, Robust)
  - Encoding (Label, One-Hot)
- Duplicate removal
- Real-time dataset statistics

###  Exploratory Data Analysis (EDA)
- Statistics Summary
- Distribution Plots
- Correlation Heatmap
- Box Plots
- Scatter Matrix
- Missing Data Visualization
- Pairwise Correlations

###  Clustering Algorithms
| Algorithm | Features |
|-----------|----------|
| **K-Means** | Elbow Method, Cluster Visualization, Silhouette Score |
| **K-Medoids** | Elbow Method, Medoid Visualization, Silhouette Score |
| **DBSCAN** | Density-based clustering, Noise detection |
| **Hierarchical (AGNES)** | Dendrogram, Multiple linkage methods |

###  Visualization
- Interactive scatter plots with cluster coloring
- PCA dimensionality reduction for high-dimensional data
- Cluster centers/medoids visualization
- Performance metrics display

---

##  How to Run

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ibtissem101/GUI-Visualisation-Machine-learning-Algos.git
   cd GUI-Visualisation-Machine-learning-Algos
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install customtkinter pandas numpy matplotlib scikit-learn scipy seaborn openpyxl
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

---

##  Project Structure

```
gui/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── ui/
│   ├── main_window.py      # Main application window
│   └── pages/
│       ├── data_loader.py  # Data loading & preprocessing
│       ├── eda.py          # Exploratory Data Analysis
│       ├── kmeans.py       # K-Means clustering
│       ├── kmedoids.py     # K-Medoids clustering
│       ├── dbscan.py       # DBSCAN clustering
│       └── hierarchical.py # Hierarchical clustering
└── README.md
```

---

##  Usage Guide

1. **Load Data:** Go to "Data Loader" and browse for your dataset file
2. **Preprocess:** Switch to "Preprocessing" tab to clean and transform your data
3. **Explore:** Use "EDA" page to visualize and understand your data
4. **Cluster:** Choose a clustering algorithm and configure parameters
5. **Visualize:** View cluster results in the Visualization tab

---

##  Technologies Used

- **CustomTkinter** - Modern UI framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib** - Plotting and visualization
- **Scikit-learn** - Machine learning algorithms
- **SciPy** - Scientific computing (for hierarchical clustering)
- **Seaborn** - Statistical visualization

---

##  Future Plans

- [ ] Add supervised learning algorithms (Classification, Regression)
- [ ] Model evaluation metrics
- [ ] Export/Save trained models
- [ ] More clustering algorithms (Gaussian Mixture, Spectral Clustering)
- [ ] Feature importance analysis

---

## 👤 Author

**Ibtissem**

---

##  License

This project is open source and available under the [MIT License](LICENSE).
