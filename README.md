# Online Retail & Electricity Demand Forecasting

Two end-to-end, production-style forecasting workflows:
- **Online Retail II** (1M+ transactions): product-level and cluster-level sales forecasting
- **Electricity Load Diagrams** (370 clients, 15-min readings): cluster-aware demand forecasting

Both projects emphasize clean data pipelines, defensible evaluation, and model comparisons across classical, deep-learning, and transformer approaches.

## 🚀 Highlights

### Retail
- Product/text clustering with TF-IDF → UMAP → MiniBatchKMeans
- Topic exploration and semantic grouping
- Forecasting via Prophet, SARIMA/ARIMA, GLM, LSTM/BiLSTM on daily/weekly series
- **Best cluster MAPE ≈ 7.65%**

### Electricity
- Rigorous chronological splits (fixes leakage)
- Savitzky–Golay smoothing for signal conditioning
- Benchmarks: BiLSTM, DeepAR (tuned), Prophet, Amazon Chronos
- **Best cluster MAPE ≈ 2.41%** (Chronos), with BiLSTM/Prophet competitive and interpretable
- 
## 📂 Repository Structure

```
.
├── Online Retail.ipynb                              # Retail EDA, preprocessing, clustering, forecasting
├── Data_Prophets_BiLstm_and_DeepAR.py              # Electricity: BiLSTM & DeepAR training/inference
├── Data_Prophets_Enhancement_TechnicalDoc...pdf    # Technical report (electricity)
├── Data_prophets_enhancement_fortune_tellers.pdf   # Slide deck (electricity)
├── Deliverable_2_Data_Prophets.pptx                # Retail deliverable slides
├── cluster_1.csv                                    # Retail cluster-level series (sample)
├── cluster_7.csv                                    # Retail cluster-level series (sample)
└── README.md
```

## 🧱 Datasets

### Online Retail II (UCI)
- **1,067,371 transactions** (2009–2011) from UK giftware retailer
- Cleaned to ~779k rows after strict preprocessing (removed cancellations, non-positive qty/price, duplicates, invalid stock codes)
- **Target**: daily/weekly `TotalSales`

### Electricity Load Diagrams 2011–2014 (UCI)
- **370 customers**, 15-minute cadence readings
- Derived daily/weekly aggregates
- Removed inconsistent 2011 data and clients with missing full years
- Final modeling set built with cluster assignments

## 🔧 Methods

### Retail Pipeline
1. **Preprocessing/EDA**: Handle cancellations, outliers, stationarity checks; create `TotalSales` feature; daily/weekly resampling
2. **Clustering**: Clean descriptions → TF-IDF → UMAP → MiniBatchKMeans → 12 semantic product groups (e.g., "Cozy Essentials & Warmers", "Baking & Party Essentials")
3. **Forecasting**: Per-cluster and global models (Prophet, ARIMA/SARIMA/SARIMAX, GLM, LSTM/BiLSTM) with rolling validation and MAPE/MAE/RMSE metrics

### Electricity Pipeline
1. **Signal Conditioning**: Savitzky–Golay smoothing; seasonality/trend decomposition; PSD/ACF diagnostics
2. **Evaluation Hygiene**: Fixed chronological train/test splits; rolling-origin validation; corrected prior leakage in DeepAR baselines
3. **Models**: BiLSTM, DeepAR (tuned), Prophet, Amazon Chronos; cluster-wise training and comparison

## 📈 Key Results (MAPE)

### Retail (Cluster-level)
- **Best cluster**: BiLSTM/Prophet achieving ~7.65% MAPE
- Other clusters: ~8–12% depending on seasonality and promotional spikes

### Electricity
| Model | Best Cluster MAPE | Notes |
|-------|-------------------|-------|
| **Amazon Chronos** | **2.41%** | Best overall performance |
| **Prophet** | 3.6–4.8% | Stable and interpretable across clusters |
| **BiLSTM** | 2–6% (median) | Strong accuracy, improved over LSTM |
| **DeepAR (tuned)** | ~2.7% | After split fix and hyperparameter search |

## ▶️ Quickstart

### Environment Setup

```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Retail Notebook

```bash
# Open Jupyter and run steps: EDA → clustering → forecasting
Jupyter Notebook "Online Retail.ipynb"
```

### Electricity Scripts

```bash
# Train BiLSTM/DeepAR per cluster (edit paths in script header)
python Data_Prophets_BiLstm_and_DeepAR.py --cluster 0 --horizon 7
```

## 🔍 Reproducibility Notes

- **Time splits**: All electricity experiments use non-overlapping, chronological splits; rolling-origin evaluation for multi-step forecasts
- **Smoothing**: Savitzky–Golay used only on inputs to reduce high-frequency noise; raw vs smoothed comparisons included in reports
- **Clusters**: 
  - Retail clusters derived from text semantics (TF-IDF embeddings)
  - Electricity clusters from multi-method consensus (K-Means/GMM/K-Shape + co-association)

## 📜 Reports & Slides

- **Retail Deliverable** (`Deliverable_2_Data_Prophets.pptx`): Methodology, EDA, cluster taxonomy, cluster-wise forecasts & errors
- **Electricity Technical Doc** (PDF): Corrected evaluation, model additions (Prophet/Chronos), tuned DeepAR, diagnostics
- **Electricity Slides** (PDF): Summary of improvements, visuals (ACF/PSD, decomposition, boxplots), model comparisons

## 📌 Roadmap

### Retail
- [ ] Add promotion/holiday exogenous features
- [ ] Implement hierarchical reconciliation (item → cluster → total)

### Electricity
- [ ] Re-clustering with usage-shape patterns + metadata
- [ ] Explore transformer families (Informer, Autoformer, PatchTST) with longer horizons

## ✍️ Citation / Attribution

Please reference the accompanying reports when using results or methodology from this repository:

- **Retail**: *Deliverable 2 – Data Prophets*
- **Electricity**: *Enhancement Technical Documentation* and *Project Improvements & Value Addition* slides

## 📦 Dependencies

Create a `requirements.txt` with the following (adjust versions as needed):

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
umap-learn>=0.5.3
matplotlib>=3.6.0
seaborn>=0.12.0
prophet>=1.1.0
statsmodels>=0.14.0
torch>=2.0.0
gluonts>=0.13.0
chronos>=0.1.0
jupyter>=1.0.0
```

## 📄 License

This project is provided for educational and research purposes. Please make sure that the original dataset licenses (UCI Machine Learning Repository) are followed.
