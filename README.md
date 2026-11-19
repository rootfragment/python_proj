# IoT Network Attack Detection: NumPy + Matplotlib Mini Project

A small machine learning project that explores **NumPy** and **Matplotlib** by classifying network traffic as **normal** or **attack** using the RT-IoT2022 dataset.

## Goal

Can very simple models tell normal IoT traffic apart from attack traffic using just 10 measurements per network flow?

## Dataset

**RT-IoT2022** is a collection of network flows from an IoT testbed (smart devices plus simulated attacks).

- 123,117 rows and 85 columns, with no missing values
- Each row is one network flow
- The label column is `Attack_type`

| Group | Types |
|---|---|
| Normal | `MQTT_Publish`, `Thing_Speak`, `Wipro_bulb` |
| Attack | `DOS_SYN_Hping`, `ARP_poisioning`, `NMAP_*` scans, `DDOS_Slowloris`, `Metasploit_Brute_Force_SSH` |

The data is heavily imbalanced: about 110,600 attack flows against about 12,500 normal flows.

## Project files

```
.
├── rt_iot2022_mini_project.py      # the full project as a script
├── RT_IOT2022                      # dataset (CSV with no extension), you add this
└── README.md
```

## Requirements

- Python 3.8 or newer
- numpy
- pandas
- matplotlib
- scikit-learn

```bash
pip install numpy pandas matplotlib scikit-learn
```

```bash
python3 rt_iot2022_mini_project.py
```

Plots open in separate windows. Close each one to move on to the next step.

To use the notebook instead, open `rt_iot2022_mini_project.ipynb` in Jupyter, VS Code or Google Colab and choose **Run All**.

If your dataset file has a different name (for example `RT_IOT2022.csv`), change this line near the top of the script:

```python
DATA_PATH = "RT_IOT2022"
```

## What the project does

1. **Load** the data with pandas.
2. **Explore** the class counts with a log-scale bar chart.
3. **Create a binary label**: `0` = normal, `1` = attack.
4. **Select 10 features**: flow duration, forward and backward packet counts, packets per second, forward and backward payload totals, SYN and RST flag counts, and forward and backward initial window sizes.
5. **Visualize**: histograms (normal vs attack), a boxplot, a scatter plot and a correlation heatmap computed with `np.corrcoef`.
6. **Preprocess with NumPy**:
   - balanced sample of 5,000 normal and 5,000 attack flows
   - shuffle with `np.random.permutation`
   - log transform (`log10(x + 1)`) to handle skewed values
   - 80/20 train/test split
   - normalization using the training mean and standard deviation only
7. **k-Nearest Neighbors from scratch** using vectorized NumPy distance calculations.
8. **Evaluate by hand**: confusion matrix, accuracy, precision, recall and F1, all computed with NumPy.
9. **Compare with scikit-learn**: k-NN, a Decision Tree (max depth 5) and Logistic Regression.
10. **Plot** accuracy against k, a feature importance chart, confusion matrices and a final model comparison.

## Results

Results from one run with `np.random.seed(42)` on a 2,000-flow test set:

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| k-NN from scratch (k=5) | 98.6% | 98.4% | 98.8% |
| scikit-learn k-NN (k=5) | 98.6% | 98.4% | 98.8% |
| Decision Tree (depth 5) | 98.5% | 98.0% | 98.9% |
| Logistic Regression | 95.8% | 98.5% | 93.0% |

## Key findings

- Attack traffic differs clearly from normal IoT traffic in packet rates, flag counts and connection behavior.
- Simple models reach about 98% accuracy, so deep learning is not needed for this dataset.
- The from-scratch NumPy k-NN matches scikit-learn exactly.
- Logistic Regression has lower recall, so it misses more attacks than the other two models.
- Accuracy alone can mislead on imbalanced data, so the project balances the sample and reports precision, recall and F1.

## Limitations

- The data comes from a single controlled testbed, so results may not carry over to real networks.
- Only 10 of the 85 features are used.
- One random sample and one train/test split are used, so numbers can vary slightly between runs.
- The binary framing does not show which specific attack was detected, and rare classes such as `Metasploit_Brute_Force_SSH` (37 flows) get little weight.
