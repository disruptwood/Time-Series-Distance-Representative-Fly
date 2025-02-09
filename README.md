# Behavior Analysis Project

This project analyzes animal behavior from `.mat` files using different computational approaches. The project involves data loading, distance computation, and clustering to find representative behavioral patterns.

---
## 📚 Used Libraries and Their Purpose

### Core Libraries
- `sys`, `os` – System utilities for file handling.
- `numpy` – Numerical operations and matrix handling.
- `pandas` – Data processing and manipulation.
- `pickle` – Serialization of objects.

### File Handling
- `scipy.io.loadmat` – Loads `.mat` files.
- `mat73` – Reads MATLAB v7.3 `.mat` files.
- `glob` – Finds files matching a pattern.
- `logging` – Logs runtime information.

### Data Processing & Analysis
- `matplotlib.pyplot` – Plots and visualizes data.
- `seaborn` – Enhances data visualization.
- `matplotlib.ticker` – Customizes plot axis ticks.
- `random` – Random selection for experiments.

### Clustering & Distance Computation
- `scipy.cluster.hierarchy` – Hierarchical clustering (dendrograms).
- `scipy.spatial.distance` – Computes distance matrices.
- `sklearn.cluster.KMeans` – K-means clustering.
- `sklearn.manifold.MDS` – Multi-dimensional scaling for visualization.
- `fastdtw` – Dynamic Time Warping for time series.
- `hmmlearn.hmm` – Hidden Markov Model-based sequence analysis.
- `scipy.special.rel_entr` – KL divergence for probability comparisons.

---
# Data Processing Overview

## 1. Data Collection and Organization
The project processes behavior classification data stored in `.mat` files. The `gather_mat_files_multiple.py` script scans the selected top-level folders recursively, searching for files that match the pattern `scores_*.mat` inside subfolders. 

- Each subfolder containing these files is treated as a separate condition.
- The condition name is derived from the subfolder name.
- The script returns:
  - `all_condition_files` → List of `.mat` files grouped by condition.
  - `condition_names` → List of detected condition names.

## 2. Data Extraction from `scores_*.mat` Files
Each `.mat` file contains an `allScores` struct with the following fields:

- `scores`: A continuous classifier output stored as a cell array, where:
  - `allScores.scores{i}(t)` represents the classifier's confidence at frame `t` for animal `i`.
  - Values are positive if the behavior is detected, negative if not.
  - `NaN` is assigned for frames where no tracking data is available.

- `tStart` & `tEnd`: Arrays storing the first and last tracked frames for each animal.

- `postprocessed`: A binary classification array, where:
  - `1` indicates the animal is performing the behavior.
  - `0` means the behavior is absent.
  - `NaN` represents frames with missing data.

- `postprocessparams`: Stores parameters used for converting raw scores into binary classifications.

- `t0s` & `t1s`: Lists containing the start (`t0s`) and end (`t1s`) frames of behavior bouts.

- `scoreNorm`: A normalization factor that scales classifier scores into the range [-1, 1].

---
## 3. Data Transformation for Processing
The extracted data is converted into multiple formats depending on the analysis approach:

### 3.1. (T, B) Binary Arrays
- Source: `postprocessed` field.
- Format: `(T, B)` array, where:
  - T = Number of time frames.
  - B = Number of behaviors.
- Usage: Used for multi-hot encoding distance calculations.

### 3.2. Behavior Intervals
- Source: `t0s` and `t1s` fields.
- Format: A list of interval dictionaries for each fly:
  ```python
  [{"tStart": X, "tEnd": Y}, {"tStart": A, "tEnd": B}, ...]
  ```
- Usage: Used for interval-based analysis and feature extraction.

### 3.3. Raw Scores (T, B)
- Source: `scores` field.
- Format: A (T, B) continuous-valued matrix.
- Usage: Used for DTW and Euclidean-based distance calculations.

---

## 4. Input Data Types for Distance Calculation Approaches
Each distance computation method processes a different data format:

| Distance Approach | Input Format | Description |
|------------------|-------------|-------------|
| Multi-Hot Encoding (Approach 1) | `(T, B)` binary array | Computes Weighted Hamming Distance between behavior sequences. |
| Interval-Based (Approach 2) | List of `{tStart, tEnd}` dictionaries | Converts intervals into feature vectors and computes Euclidean distance. |
| Hidden Markov Model (Approach 3) | `(T, B)` continuous-valued matrix | Fits an HMM to behavioral data and computes KL divergence between models. |
| Dynamic Time Warping (Approach 4) | `(T, B)` continuous-valued matrix | Uses DTW to compare behavior time series. |
| Simple Euclidean Distance (Approach 5) | `(T, B)` continuous-valued matrix | Directly computes Euclidean distance between raw scores. |

Each method is selected based on the required level of abstraction and sensitivity to time variations in behavior sequences.



---
## 🔧 How to Use the Interface

1. Run `main.py` – This script guides the user through selecting input data and processing it.

2. Choose Input Mode:
   - `single` – Select `.mat` files for one condition using a file selection dialog.
   - `multiple` – Drag and drop folders containing `.mat` files into the selection window.

3. Choose Analysis Approach:
   - Approach 1: Multi-hot encoded behavior distances.
   - Approach 2: Behavioral interval comparison.
   - Approach 3: Hidden Markov Model-based analysis.
   - Approach 4: Dynamic Time Warping on classifier scores.
   - Approach 5: Simple Euclidean distance on scores.

4. Select Number of Representatives – The script finds the most representative samples based on distance computations.

---

## 📊 `DataAnalysis.ipynb`
The `DataAnalysis.ipynb` notebook was used for research and data exploration. It contains exploratory analysis, visualizations, and validation of different distance computation methods.

---

## Running the Project
Make sure you have all required dependencies installed before running the project. Use:

pip install numpy pandas scipy sklearn matplotlib seaborn fastdtw hmmlearn
Then, execute:
"python main.py"
'Follow the interface prompts to process behavioral data and analyze results.'
