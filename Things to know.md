# 📚 IQUEUE — Machine Learning Pipeline Documentation

Everything done in this project mapped to the standard ML process.

---

## Step 1: Data Collection

**Description:** Gathering relevant raw data from various sources.

### What We Did
Since no real historical queue logs exist from LTO CDO, we **generated synthetic data** that realistically simulates LTO CDO queue behavior. This is a valid approach in research when real data is unavailable — the synthetic data is built on real-world logic and domain knowledge about how government queues work.

### Tool: `data/Data_.py`
- Simulates **13 weeks** of queue transactions (Jan 1 – Mar 31, 2025)
- Covers **Monday to Saturday**, hours **8:00 AM – 4:00 PM**
- Generates **8–14 random customer arrivals per hour** per day
- Total records generated: **6,498 transactions**

### Ground Truth Patterns Used (Domain Knowledge)
These are the real-world-informed average wait times (in minutes) baked into the simulation:

| Hour | Mon | Tue | Wed | Thu | Fri | Sat |
|------|-----|-----|-----|-----|-----|-----|
| 8am  | 25  | 12  |  9  | 11  | 22  | 14  |
| 9am  | 55  | 28  | 19  | 28  | 50  | 28  |
| 10am | 70  | 32  | 25  | 32  | 65  | 33  |
| 11am | 60  | 25  | 18  | 26  | 55  | 25  |
| 12pm | 25  | 15  |  9  | 16  | 26  | 20  |
| 1pm  | 50  | 20  | 15  | 20  | 48  | 24  |
| 2pm  | 65  | 30  | 22  | 30  | 60  | 28  |
| 3pm  | 55  | 25  | 18  | 24  | 51  | 25  |
| 4pm  | 35  | 15  | 12  | 15  | 32  | 19  |

Monday and Friday are peak days. Wednesday is the lightest day.

### Realistic Noise Added
Raw patterns are not used directly — ±15% random variation is added per transaction:
```python
wait_variation = np.random.uniform(0.85, 1.15)
wait_time = base_wait * wait_variation
wait_time = max(5, min(90, wait_time))   # capped between 5–90 min
```

### Output
Saved to: `data/synthetic_lto_cdo_queue_90days.csv`

---

## Step 2: Data Preparation

**Description:** Cleaning, transforming, and formatting data — handling missing values, encoding categorical variables, normalizing features.

### Tool: `src/preprocess.py` + `data/Data_.py`

### 2a. Cleaning (`preprocess.py → load_data()`)
- Parses `date` column into proper datetime format
- Removes rows where `waiting_time_min < 0` (invalid)
- Removes rows where `queue_length_at_arrival < 0` (invalid)
- Drops any rows with missing values (`dropna()`)

### 2b. Encoding Categorical Variables
Day names (text) are encoded as numbers so the model can process them:

| Day Name  | `day_of_week` Encoding |
|-----------|------------------------|
| Monday    | 0                      |
| Tuesday   | 1                      |
| Wednesday | 2                      |
| Thursday  | 3                      |
| Friday    | 4                      |
| Saturday  | 5                      |

Binary flags are also created:
- `is_peak_day` → 1 if Monday or Friday, else 0
- `is_peak_hour` → 1 if hour in [9,10,11,14,15], else 0
- `is_weekend` → 1 if Saturday, else 0

### 2c. Feature Engineering
New meaningful columns derived from existing data (no new CSV needed):

**`week_of_month`** — computed from the `date` column at runtime:
```python
df['week_of_month'] = df['date'].dt.day.apply(lambda d: (d - 1) // 7 + 1)
```
```
Day  1–7  → Week 1
Day  8–14 → Week 2
Day 15–21 → Week 3
Day 22–31 → Week 4
```
This allows Monday April 7 (Week 1) to differ from Monday April 28 (Week 4).

**Lag Features** — captures the previous transaction's context within the same day:
```python
df['queue_length_lag1'] = df.groupby('date')['queue_length_at_arrival'].shift(1)
df['waiting_time_lag1'] = df.groupby('date')['waiting_time_min'].shift(1)
```
For the **first transaction of each day**, defaults are:
- Peak days (Mon/Fri): `queue_lag=8`, `wait_lag=25`  
- Other days: `queue_lag=3`, `wait_lag=10`

### 2d. Final Feature Set (10 inputs, 1 target)

| Role | Column | Type |
|------|--------|------|
| **Target (y)** | `waiting_time_min` | Continuous (minutes) |
| Feature 1 | `hour` | Numeric (8–16) |
| Feature 2 | `day_of_week` | Encoded (0–5) |
| Feature 3 | `week_of_month` | Engineered (1–4) |
| Feature 4 | `is_peak_day` | Binary (0/1) |
| Feature 5 | `queue_length_at_arrival` | Numeric |
| Feature 6 | `service_time_min` | Numeric |
| Feature 7 | `is_weekend` | Binary (0/1) |
| Feature 8 | `is_peak_hour` | Binary (0/1) |
| Feature 9 | `queue_length_lag1` | Numeric |
| Feature 10 | `waiting_time_lag1` | Numeric |

> Note: No normalization/scaling is needed for Random Forest — tree-based models are scale-invariant.

---

## Step 3: Data Segregation

**Description:** Splitting the dataset into training, validation, and test sets.

### Tool: `src/train_model.py`

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

| Split | Proportion | Records (approx.) | Purpose |
|-------|------------|-------------------|---------|
| **Training set** | 80% | ~5,198 rows | Model learns patterns from this |
| **Test set** | 20% | ~1,300 rows | Evaluate model on unseen data |

- `random_state=42` ensures the same split every run (reproducible)
- No separate validation set — cross-validation is implicitly handled by Random Forest's internal bootstrapping (bagging)

---

## Step 4: Training of Model

**Description:** Model learns patterns and relationships from the training data.

### Tool: `src/train_model.py`
### Algorithm: Random Forest Regressor

A **Random Forest** trains hundreds of independent decision trees on random subsamples of the training data, then averages their predictions. This reduces overfitting and handles non-linear relationships well.

```python
model = RandomForestRegressor(
    n_estimators=500,       # 500 decision trees built
    max_depth=15,           # each tree can go 15 levels deep
    min_samples_split=5,    # node must have ≥5 samples to split further
    min_samples_leaf=2,     # leaf nodes must contain ≥2 samples
    max_features='sqrt',    # each tree sees √10 ≈ 3 features per split
    random_state=42         # reproducible results
)

model.fit(X_train, y_train)
```

### What the Model Learned (Feature Importances)

| Rank | Feature | Importance | Insight |
|------|---------|------------|---------|
| 1 | `waiting_time_lag1` | 33.4% | Previous wait is the strongest predictor |
| 2 | `queue_length_at_arrival` | 24.1% | Current queue size is key |
| 3 | `service_time_min` | 16.5% | Transaction duration drives wait |
| 4 | `queue_length_lag1` | 13.0% | Momentum from previous queue matters |
| 5 | `is_peak_day` | 6.2% | Mon/Fri flag adds clear signal |
| 6 | `hour` | 2.3% | Time of day contributes |
| 7 | `is_peak_hour` | 2.3% | Peak hour flag contributes |
| 8 | `day_of_week` | 1.8% | Specific day refines prediction |
| 9 | `week_of_month` | 0.22% | Adds week-level date variation |
| 10 | `is_weekend` | 0.19% | Saturday flag has minor effect |

### Saved Model
- Path: `models/queue_model.pkl`
- Size: ~66 MB
- Saved using `joblib.dump()`, loaded using `joblib.load()`

---

## Step 5: Model Evaluation

**Description:** Assessing model performance using metrics.

### Tool: `src/train_model.py` → results in `outputs/metrics.txt`

The model was evaluated on the **held-out test set** (20% of data it never trained on):

| Metric | Value | What It Means |
|--------|-------|---------------|
| **MAE** (Mean Absolute Error) | 2.25 min | On average, predictions are off by ~2.25 minutes |
| **RMSE** (Root Mean Squared Error) | 3.14 min | Slightly penalizes larger errors more than MAE |
| **R² Score** | **0.9648** | Model explains **96.5% of the variance** in wait times |

### Extended Evaluation (Recommended and now implemented in `src/train_model.py`)

To avoid relying only on MAE/RMSE/R², the pipeline now computes additional diagnostics in `outputs/metrics.txt`.

### Current Extended Results (from latest run)

| Check | Current Value | Interpretation |
|------|---------------|----------------|
| **Baseline MAE improvement** | **83.73%** | Model is far better than naive mean prediction |
| **Baseline RMSE improvement** | **81.26%** | Strong gain over baseline on larger errors |
| **Overfitting gap (MAE test - train)** | **0.82 min** | Expected gap; not extreme |
| **Overfitting gap (R² train - test)** | **0.0213** | Good generalization with small drop on test |
| **Tail error (P90 / P95 / Max AE)** | **5.41 / 6.94 / 14.58 min** | Most errors stay moderate; worst-case still tracked |
| **Chronological split MAE / RMSE / R²** | **2.35 / 3.31 / 0.9620** | Time-aware validation remains strong |
| **Uncertainty coverage (P10-P90)** | **65.77%** | Actual waits inside predicted band ~66% of time |
| **Average uncertainty band width** | **5.81 min** | Typical P10-P90 spread |
| **Peak day MAE vs non-peak day MAE** | **3.45 vs 1.48 min** | Peak-day predictions are harder |
| **Peak hour MAE vs non-peak hour MAE** | **2.76 vs 1.57 min** | Busy hours increase error |

### Why These Evaluations Were Added

Each added evaluation answers a different reliability question that MAE/RMSE/R² alone cannot fully answer:

1. **Baseline comparison** — Reason: proves the model is genuinely useful and not just matching what a simple average predictor can do.
2. **Overfitting check (train vs test gaps)** — Reason: verifies the model generalizes to unseen data instead of memorizing training patterns.
3. **Tail-error metrics (P90, P95, Max AE)** — Reason: captures high-impact mistakes during worst-case conditions, which averages can hide.
4. **Time-aware validation (chronological split)** — Reason: simulates real deployment where we predict future dates from past data.
5. **Segment-wise errors (day/hour/peak flags)** — Reason: identifies operational weak spots (for example, peak windows) so improvements can be targeted.
6. **Uncertainty coverage and band width** — Reason: checks whether forecast ranges are trustworthy, not just point predictions.

These diagnostics are generated automatically every time `src/train_model.py` runs.

### Interpretation
- **R² of 0.96** is considered excellent. A perfect model = 1.0, a model that just guesses the average = 0.0.
- **MAE of 2.25 min** means if the model says "43 minutes," the true wait will likely be between 41–45 minutes.
- These metrics were computed on **unseen test data** — so they reflect how the model performs on new inputs, not just what it memorized.

> Note: Since this is a **regression** problem (predicting a number, not a category), metrics like accuracy, precision, recall, and F1 are not applicable. For this project, MAE/RMSE/R² are core metrics, and baseline, segmentation, tail-error, time-aware, and uncertainty checks are complementary diagnostics.

---

## Step 6: Model Deployment

**Description:** Deploying the final trained model into a real-world environment.

### Current Deployment: CLI Application (`src/predict.py`)

The trained model is loaded and served through an interactive command-line interface:

```python
model = joblib.load("models/queue_model.pkl")  # load once at startup
```

Users interact via a menu:
1. **Weekly forecast** — enter any date, see all 6 days for that week
2. **Specific date forecast** — hourly congestion breakdown for a date
3. **Best time finder** — identifies the lowest and highest wait hour for a date

### Prediction Flow at Runtime
```
User enters date (e.g., 2026-04-07)
        ↓
Extract: day_name = "Monday", week_of_month = 1
        ↓
Look up historical averages:
  queue_patterns['Monday'][1][hour]  ← Week-1 Monday avg queue
  wait_patterns['Monday'][1][hour]   ← Week-1 Monday avg wait (for lag)
        ↓
Build 10-feature vector
        ↓
Generate N randomized feature variants (Monte Carlo)
        ↓
model.predict(X_simulated)  → distribution of wait times
        ↓
Summarize uncertainty: Mean, P10, P50, P90
        ↓
Display: congestion level + recommendation + likely range
```

### Monte Carlo Improvement in Deployment

The CLI now runs a **Monte Carlo uncertainty simulation** in `src/predict.py` for each hourly forecast:

- Uses **1,000 simulation runs** per hour (`MONTE_CARLO_RUNS = 1000`)
- Perturbs operational inputs (queue length, service time, lag features)
- Produces a prediction distribution, not just one point estimate
- Displays:
  - **Mean** predicted wait (used for congestion label)
  - **P10-P90 likely range** (uncertainty band)

This means users now see both the expected value and a realistic uncertainty range, which is more useful for planning.

### Congestion Thresholds Applied Post-Prediction
```
Mean predicted wait > 45 min  →  🔴 HIGH      — ❌ AVOID
Mean predicted wait > 25 min  →  🟡 MODERATE  — ⚠️ CAUTION
Mean predicted wait ≤ 25 min  →  🟢 LOW       — ✅ GOOD
```

### What This System Is (and Is Not)
| ✅ This system IS | ❌ This system is NOT |
|------------------|----------------------|
| A **planning tool** — helps users pick the best date/time to visit | A **real-time tracker** — cannot tell current queue length |
| Based on **historical patterns** from 90 days of data | Predicting a **specific person's exact wait** when already in queue |
| Reliable for **day-of-week and week-of-month** trends | Accounting for **today's unexpected events** (holidays, staff absences) |
| Includes **uncertainty-aware forecasts** via Monte Carlo ranges (P10-P90) | A guarantee that wait time will exactly match one number |

### Simulation Scope Clarification

- `data/Data_.py` simulates and generates the **training dataset** (historical synthetic transactions).
- `src/predict.py` simulates **prediction uncertainty** at runtime using Monte Carlo.
- Both are valid, but they solve different problems:
        - Data generation simulation -> model training input
        - Monte Carlo simulation -> forecast confidence/range output

---

## 📁 Project File Map

```
IQUEUE/
├── data/
│   ├── Data_.py                           ← Step 1: Generates synthetic data
│   └── synthetic_lto_cdo_queue_90days.csv ← Raw collected data (6,498 rows)
├── models/
│   └── queue_model.pkl                    ← Step 4–5: Trained & evaluated model
├── outputs/
│   └── metrics.txt                        ← Step 5: Core + extended evaluation diagnostics
├── src/
│   ├── preprocess.py                      ← Step 2: Data preparation & feature engineering
│   ├── train_model.py                     ← Steps 3–5: Segregation, training, evaluation
│   ├── predict.py                         ← Step 6: Deployment + Monte Carlo uncertainty forecast
│   └── utils.py                           ← Utilities
├── main.py                                ← Entry point
└── Things to know.md                      ← This file
```
