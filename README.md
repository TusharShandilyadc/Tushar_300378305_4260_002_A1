# Tushar_300378305_4260_002_A1
# README - Data Benchmarking and Machine Learning Pipeline

## Project Overview
This assignment focuses on benchmarking the efficiency of CSV and Parquet file formats, comparing Pandas and Polars for data processing, and implementing machine learning models (**XGBoost and Gradient Boosting Regressor (GBR)**) for stock price prediction. 

The objective is to analyze file storage, read/write performance, and computational efficiency while ensuring a structured approach to model selection and predictive modeling.

## Why This Approach?
### 1. **File Format Benchmarking**
   - **CSV vs. Parquet**: CSV is a widely used data storage format but lacks efficiency when handling large datasets. Parquet, on the other hand, is optimized for columnar storage, leading to better compression and faster read/write operations.
   - **Compression Choice**: Instead of using Snappy, which is the default Parquet compression, I used **Gzip** to analyze the impact of different compression methods on file size and processing speed.
   - **Scalability Testing**: I created **10x and 100x versions** of both CSV and Parquet files to examine performance across different dataset sizes.
   - **Performance Benchmarking**: Read/write times and file sizes were recorded and stored in `benchmark_results.csv`, providing insights into the storage efficiency and I/O performance of each format.

### 2. **Pandas vs. Polars Comparison**
   - **Why Compare?** Pandas is the standard tool for data analysis in Python, but Polars is designed to be faster and more memory-efficient, especially for large datasets.
   - **Performance Evaluation**: I loaded and processed data using both Pandas and Polars, measuring execution times to determine which library performed better in handling large-scale stock market data.
   - **Findings**: Polars demonstrated superior speed, particularly in reading and manipulating large files, proving beneficial for data-intensive operations. The results are stored in `pandas_vs_polars.csv`.

### 3. **Machine Learning Models Selection**
   - **Why XGBoost and GBR?** I experimented with different models and found that **Gradient Boosting Regressor (GBR) significantly outperformed XGBoost** for this dataset.
   - **XGBoost**: Known for its efficiency and optimized performance in structured data, XGBoost was included for comparison.
   - **GBR (Gradient Boosting Regressor)**: It performed better due to its ability to capture complex patterns in the data, providing lower error rates.
   - **Feature Selection**: Instead of relying on technical indicators like SMA, EMA, RSI, and MACD for model training, I used the raw dataset to ensure model performance was evaluated on unprocessed data.
   - **Model Outputs**: Both models were trained, evaluated, and stored as `xgb_model.pkl` and `gbr_model.pkl` for later use.
   - **Standardization**: A **StandardScaler** was used to normalize features, and the scaler was saved separately as `scaler.pkl` to ensure consistency in future model predictions.

## Setup Instructions
### Step 1: Create a Virtual Environment
To ensure dependency management, create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

### Step 2: Install Required Packages
Run:

```bash
pip install -r requirements.txt
```


### Step 3: Running the Project
#### Option 1: Running Jupyter Notebook (Recommended but Not Mandatory)
If you want to run the Jupyter Notebook to see the benchmarking and model training:

```bash
jupyter notebook
```

Open `Stock_Price_Prediction_Research` and run all cells.

#### Option 2: Skipping Jupyter Notebook (Direct Streamlit Execution)
If you do not want to run the Jupyter Notebook, make sure to download or clone the repository and place the following files in the **same folder**:
- `benchmark_results.csv`
- `pandas_vs_polars.csv`
- `xgb_model.pkl`
- `gbr_model.pkl`
- `scaler.pkl`

Then, create a virtual environment as explained in **Step 1**, install dependencies (**Step 2**), and directly run the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

## Key Learnings and Outcomes
### 1. **CSV vs. Parquet Trade-offs**
   - Parquet files were significantly smaller in size due to efficient columnar compression.
   - Read and write operations were **faster for Parquet** than CSV, especially with larger datasets (10x and 100x versions).
   - CSV format was simpler but inefficient for large-scale storage and analysis.

### 2. **Pandas vs. Polars Performance**
   - Polars was **faster in reading large datasets**, making it a better choice for handling extensive stock market data.
   - Pandas remains the dominant library but struggles with performance as dataset size increases.

### 3. **Machine Learning Model Comparison**
   - **GBR outperformed XGBoost** in terms of accuracy and overall error reduction.
   - XGBoost is still a reliable choice, but GBR provided better predictions for this particular dataset.
   - Feature selection was based on raw data rather than technical indicators to maintain generalizability.

This structured approach ensures a well-rounded analysis of storage, computation, and machine learning in real-world datasets.

