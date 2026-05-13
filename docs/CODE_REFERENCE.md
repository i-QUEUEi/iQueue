# iQueue Code Reference

> Think of iQueue like a **smart restaurant host**. Before you arrive, it has already studied months of past customer patterns. When you ask "how long is the wait?", it doesn't guess — it checks what day it is, what time it is, how busy it usually gets, and gives you a real answer. This document explains how all the code pieces work together to make that happen.

---

## 🗺️ The Big Picture: What Happens When You Run the System

```
You type: python main.py
       ↓
[Step 1] Train the model → reads data → picks the best model → saves it
       ↓
[Step 2] Start predictions → loads saved model → you ask questions → it answers
```

There are **two major phases**:
1. **Training phase** — the system learns from past data
2. **Prediction phase** — you interact with what it learned

---

## 📁 File Index (Quick Map)

| File | What it does in one sentence |
|---|---|
| [main.py](../main.py) | The "start button" — runs training first, then opens the prediction app |
| [data/Data_.py](../data/Data_.py) | Generates fake-but-realistic LTO queue data for 90 days |
| [src/Preprocessing/loader.py](../src/Preprocessing/loader.py) | Cleans the raw data and adds useful columns (day, hour, holidays, etc.) |
| [src/Preprocessing/features.py](../src/Preprocessing/features.py) | Defines exactly which columns the model is allowed to see |
| [src/Preprocessing/calendar.py](../src/Preprocessing/calendar.py) | Reads a holiday list so the system knows when special days are |
| [src/Preprocessing/preprocess.py](../src/Preprocessing/preprocess.py) | One import that brings all preprocessing steps together |
| [src/model_implementation/train_model.py](../src/model_implementation/train_model.py) | The main trainer — runs all models, picks the best one, saves it |
| [src/model_implementation/model_zoo/linear_regression.py](../src/model_implementation/model_zoo/linear_regression.py) | Model #1: Simple straight-line guesser (baseline) |
| [src/model_implementation/model_zoo/random_forest.py](../src/model_implementation/model_zoo/random_forest.py) | Model #2: Group of decision trees voting together |
| [src/model_implementation/model_zoo/gradient_boosting.py](../src/model_implementation/model_zoo/gradient_boosting.py) | Model #3: Trees that learn from each other's mistakes |
| [src/model_implementation/model_zoo/__init__.py](../src/model_implementation/model_zoo/__init__.py) | The "model menu" — lists all models to test |
| [src/Evaluation/evaluation.py](../src/Evaluation/evaluation.py) | Tests how good each model is using 3 different methods |
| [src/Evaluation/metrics.py](../src/Evaluation/metrics.py) | Calculates the actual error scores (MAE, RMSE, R²) |
| [src/Evaluation/splits.py](../src/Evaluation/splits.py) | Splits data by date order (oldest → train, newest → test) |
| [src/Evaluation/plots.py](../src/Evaluation/plots.py) | Draws 4 charts to visualize model performance |
| [src/Evaluation/reporting.py](../src/Evaluation/reporting.py) | Writes a full human-readable report explaining everything |
| [src/Evaluation/samples.py](../src/Evaluation/samples.py) | Runs a few test predictions to make sure the model "makes sense" |
| [src/Prediction/context.py](../src/Prediction/context.py) | Loads the saved model and builds a lookup table of historical patterns |
| [src/Prediction/patterns.py](../src/Prediction/patterns.py) | Reads the lookup table to find what the queue "usually looks like" at a given time |
| [src/Prediction/inference.py](../src/Prediction/inference.py) | Builds the input and asks the model for a prediction |
| [src/Prediction/cli.py](../src/Prediction/cli.py) | The menu you interact with (weekly view, daily view, best time finder) |
| [src/Prediction/predict.py](../src/Prediction/predict.py) | Entry point — sets up paths and starts the CLI |

---

## 🔁 Phase 1: Training — "How the System Learns"

> **Analogy:** Imagine you're studying for an exam. You read all your past notes (data), practice with different study strategies (models), check which strategy gave the best score (evaluation), then write down your final notes (model file + report).

### Step 1 — Start the pipeline

**File:** [main.py](../main.py)

When you run `python main.py`, two things happen in order:

1. It runs [train_model.py](../src/model_implementation/train_model.py) first — this is where the learning happens.
2. Only after training succeeds does it open the prediction app.

This order matters. If training ran second, the prediction app would try to load a model that doesn't exist yet and crash.

```python
# main.py — Line 7: Run training first
subprocess.run([sys.executable, "src/model_implementation/train_model.py"], check=True)

# main.py — Line 10: Then start predictions
subprocess.run([sys.executable, "src/Prediction/predict.py"], check=True)
```

---

### Step 2 — Generate / Load the data

**File:** [data/Data_.py](../data/Data_.py)

Since real LTO CDO queue records aren't always available, this file **simulates 90 days of realistic queue data**. It's not random noise — it's designed to match real patterns:

- Mondays and Fridays are busier (because that's how government offices work)
- 9am–11am are peak hours
- Holidays make queues spike
- Some random variation is added so the model doesn't "memorize" perfect patterns

> **Analogy:** It's like a movie prop department building a fake hospital room. It's not a real hospital, but everything looks and behaves like one so the actors (models) can practice realistically.

The output is saved to: `data/synthetic_lto_cdo_queue_90days.csv`

---

### Step 3 — Clean and prepare the data

**Files:** [loader.py](../src/Preprocessing/loader.py), [features.py](../src/Preprocessing/features.py), [calendar.py](../src/Preprocessing/calendar.py), [preprocess.py](../src/Preprocessing/preprocess.py)

Raw data is messy. This step cleans it and adds columns that make the model smarter:

| What it adds | Why |
|---|---|
| `day_of_week` (0=Mon, 5=Sat) | The model needs to know which day it is |
| `hour` (8–16) | Different hours have different wait times |
| `week_of_month` (1–4) | End-of-month is usually busier |
| `is_holiday`, `is_pre_holiday` | Holidays change queue behavior dramatically |
| `month_sin`, `month_cos` | Encodes the month cyclically (December wraps back to January) |
| `queue_length_lag1` | How long was the queue 1 transaction ago? (momentum) |
| `waiting_time_lag1` | How long did the previous person wait? (momentum) |
| `is_peak_day`, `is_peak_hour` | Quick flag for Mon/Fri and 9–11am |

Rows with negative waiting times or missing values are removed — garbage in, garbage out.

> **Analogy:** Think of this like a chef prepping ingredients. The raw chicken (data) needs to be washed, cut, and seasoned (cleaned + feature-engineered) before it goes in the oven (model).

[calendar.py](../src/Preprocessing/calendar.py) specifically reads a text file of Philippine holidays and converts them to a set of dates so the system can quickly check "is this date a holiday?"

---

### Step 4 — Test all models

**Files:** [train_model.py](../src/model_implementation/train_model.py), [model_zoo/](../src/model_implementation/model_zoo/__init__.py), [evaluation.py](../src/Evaluation/evaluation.py)

Three models compete against each other:

#### Model #1 — Linear Regression
**File:** [linear_regression.py](../src/model_implementation/model_zoo/linear_regression.py)

> **Analogy:** Drawing a straight line through all the data points. Simple, fast, and easy to explain. It works if the relationship is roughly linear (e.g. "more people in queue = longer wait"). It won't capture complex patterns though.

#### Model #2 — Random Forest
**File:** [random_forest.py](../src/model_implementation/model_zoo/random_forest.py)

> **Analogy:** Ask 100 different people for their opinion, then take the majority vote. Each "person" is a decision tree trained on slightly different data. The combined answer is more reliable than any single tree.

Handles non-linear patterns naturally (e.g. "Monday mornings are bad BUT only during the last week of the month").

#### Model #3 — Gradient Boosting
**File:** [gradient_boosting.py](../src/model_implementation/model_zoo/gradient_boosting.py)

> **Analogy:** Imagine a student who keeps reviewing only the questions they got wrong. Each new tree in gradient boosting focuses on fixing the previous tree's mistakes. It's slower to train but often more accurate.

The model "menu" that lists all three lives in [model_zoo/\_\_init\_\_.py](../src/model_implementation/model_zoo/__init__.py).

---

### Step 5 — Evaluate each model (3 different ways)

**File:** [evaluation.py](../src/Evaluation/evaluation.py)

Just checking accuracy on one test set is not enough — a model might get lucky. So each model is tested **3 ways**:

#### Way 1 — Random Split (Lines 48–49)
Randomly shuffle all data, put 80% in training and 20% in testing.
> **Analogy:** Shuffle a deck of cards and deal 20% out as your "quiz cards." The model hasn't seen them before.

#### Way 2 — Chronological Split (Lines 51–53, using [splits.py](../src/Evaluation/splits.py))
Use the **oldest 80% of dates** for training and the **newest 20%** for testing.
> **Analogy:** Study using notes from January–October, then take the November–December exam. This tests whether the model can handle future dates, not just random ones it hasn't memorized.

The chronological split logic in [splits.py](../src/Evaluation/splits.py) sorts by date and cuts at 80% of unique dates — so even if some dates have more rows than others, the split stays fair.

#### Way 3 — 5-Fold Cross-Validation (Lines 55–66)
Split the data into 5 equal chunks. Train on 4, test on 1. Repeat 5 times, rotating which chunk is the test. Average the results.
> **Analogy:** Take 5 different practice exams from different tutors. Your average score across all 5 is more reliable than any single exam.

All three results are averaged into one **`robust_mae`** score (Line 75):
```python
robust_mae = float(np.mean([test_metrics["mae"], chrono_metrics["mae"], cv_mae]))
```
The model with the **lowest `robust_mae`** wins and gets saved.

---

### Step 6 — Evaluate the data itself

**File:** [evaluation.py — Lines 137–155](../src/Evaluation/evaluation.py#L137-L155)

Before training, `evaluate_data_quality()` checks the **raw CSV** for problems:

- How many rows and columns?
- Any duplicate rows?
- Any missing values?
- Any negative waiting times (which are impossible in real life)?
- What is the average, min, max, and spread of waiting times?

This gets written to `outputs/metrics.txt` so you can audit your data quality.

---

### Step 7 — Generate charts

**File:** [plots.py](../src/Evaluation/plots.py)

Four charts are saved to `outputs/plots/`:

| Chart | File | What it shows |
|---|---|---|
| Waiting time distribution | `target_distribution.png` | Are most waits short or long? Any outliers? |
| Day × Hour heatmap | `day_hour_heatmap.png` | Which day + hour combinations are the busiest? |
| Model comparison | `model_comparison.png` | Side-by-side bar chart of all 3 models' MAE scores |
| Actual vs Predicted | `actual_vs_predicted.png` | How close are predictions to reality? Points near the diagonal line = good |

All 4 are triggered from [train_model.py Lines 113–116](../src/model_implementation/train_model.py#L113-L116).

---

### Step 8 — Write the report

**File:** [reporting.py](../src/Evaluation/reporting.py)

`write_report()` generates `outputs/metrics.txt` with these sections:

| Section | Lines | What's in it |
|---|---|---|
| DATA EVALUATION | [L11–22](../src/Evaluation/reporting.py#L11-L22) | Row count, duplicates, missing values, wait time stats |
| MODEL BENCHMARK | [L24–34](../src/Evaluation/reporting.py#L24-L34) | All 3 models ranked by MAE/RMSE/R² |
| BASELINE COMPARISON | [L36–39](../src/Evaluation/reporting.py#L36-L39) | What score a "just guess the average" model would get |
| ROBUST EVALUATION | [L41–52](../src/Evaluation/reporting.py#L41-L52) | The winning model's full test results (random + chrono + percentile errors) |
| SEGMENT ERROR CHECKS | [L54–58](../src/Evaluation/reporting.py#L54-L58) | Is the model worse on peak days/hours vs normal ones? |
| WHY THESE MODELS | [L60–63](../src/Evaluation/reporting.py#L60-L63) | Plain-English justification for each model chosen |
| WHY NOT OTHERS | [L65–68](../src/Evaluation/reporting.py#L65-L68) | Why neural networks, scaling, one-hot encoding weren't used |
| PREPROCESSING CHOICES | [L70–73](../src/Evaluation/reporting.py#L70-L73) | Why lag features, week_of_month, and filtering were done |
| FEATURE IMPORTANCE | [L75–77](../src/Evaluation/reporting.py#L75-L77) | Which input columns matter most to the winning model |

---

### Step 9 — Sanity check predictions

**File:** [samples.py](../src/Evaluation/samples.py)

After saving the model, 6 hardcoded test cases are run through it (Lines 7–14). These are scenarios where you already know the expected answer, like:

- "Monday 9am with 25 people in queue → should be around 55 min"
- "Wednesday 8am with 4 people → should be around 9 min"

If these numbers are completely off, something went wrong in training.

---

## 🔮 Phase 2: Prediction — "How the System Answers Your Questions"

> **Analogy:** A trained doctor (the model) has finished studying (training). Now patients walk in (you ask questions) and the doctor gives advice based on everything they learned.

### Step 1 — Load the model and build pattern maps

**File:** [context.py](../src/Prediction/context.py)

When the prediction app starts, it:
1. Loads the saved model from `models/queue_model.pkl`
2. Reads the original CSV data
3. Builds a **lookup table** of historical patterns organized by: month → week-of-month → day → hour

> **Analogy:** The doctor not only uses their knowledge (model) but also keeps a reference chart on the wall: "On average, Monday mornings in week 1 of January look like THIS."

This lookup table is used to fill in queue-related features when making predictions.

---

### Step 2 — Look up historical patterns

**File:** [patterns.py](../src/Prediction/patterns.py)

When you ask about a specific date and time, this file checks:
- What month is it? → What week of the month? → What day? → What hour?
- Then returns the average queue length and average wait time from real historical data for that slot.

If the exact slot has no data (e.g. the data only covers 90 days), it falls back to a broader average — so it always returns something reasonable.

---

### Step 3 — Build inputs and predict

**File:** [inference.py](../src/Prediction/inference.py)

This is where the actual prediction happens. For each hour you're asking about:

1. It assembles a row of features (day, hour, holiday flags, queue size from patterns, etc.) — the exact same format the model was trained on.
2. It passes that row to the ML model.
3. The model outputs a predicted wait time in minutes.

It also runs **Monte Carlo simulation** — it runs the prediction 1,000 times with small random variations, then gives you a confidence range (P10 to P90).

> **Analogy:** Instead of saying "the wait is exactly 32 minutes," it says "it's usually between 28 and 36 minutes" — like a weather forecast saying "60% chance of rain."

---

### Step 4 — Show results to the user

**File:** [cli.py](../src/Prediction/cli.py)

This is the menu interface you see in the terminal. It has 3 options:

| Option | What it does |
|---|---|
| Weekly forecast | Shows all 6 days of a chosen week with best/worst times |
| Specific date | Shows hour-by-hour breakdown for one date |
| Best time finder | Tells you the single best hour to visit on a given date |

The results use emoji-coded congestion levels:
- 🟢 LOW — under 25 min
- 🟡 MODERATE — 25–45 min
- 🔴 HIGH — over 45 min

---

## 📐 Error Metrics Explained Simply

| Metric | What it means | Good value |
|---|---|---|
| **MAE** (Mean Absolute Error) | On average, how many minutes off is the prediction? | Lower = better. Under 10 min is great. |
| **RMSE** (Root Mean Squared Error) | Like MAE but punishes big mistakes more | Lower = better |
| **R²** (R-squared) | How much of the wait time variation does the model explain? | 1.0 = perfect, 0 = no better than guessing the average |
| **P90 error** | 90% of predictions are within this many minutes of the truth | Tells you the "worst normal case" |

---

## 🔗 How the Files Connect (Data Flow)

```
data/Data_.py
   → writes: data/synthetic_lto_cdo_queue_90days.csv
        ↓
src/Preprocessing/loader.py + features.py + calendar.py
   → cleans data, adds features
        ↓
src/model_implementation/train_model.py
   → tests all 3 models using Evaluation/
   → saves best model to: models/queue_model.pkl
   → saves charts to: outputs/plots/
   → saves report to: outputs/metrics.txt
        ↓
src/Prediction/context.py
   → loads queue_model.pkl + CSV
   → builds pattern lookup table
        ↓
src/Prediction/inference.py
   → builds feature rows → asks model → gets wait time
        ↓
src/Prediction/cli.py
   → shows you weekly/daily forecasts and best times
```

---

## 📖 Recommended Reading Order

1. [main.py](../main.py) — Start here. Just 10 lines. Understand what runs and in what order.
2. [data/Data_.py](../data/Data_.py) — See how the training data is made and what patterns it simulates.
3. [src/Preprocessing/loader.py](../src/Preprocessing/loader.py) — Understand what cleaning and feature engineering is applied.
4. [src/Evaluation/evaluation.py](../src/Evaluation/evaluation.py) — See the 3-way evaluation strategy and how `robust_mae` is computed.
5. [src/model_implementation/train_model.py](../src/model_implementation/train_model.py) — The full training pipeline in one place.
6. [src/Prediction/inference.py](../src/Prediction/inference.py) — How a prediction is made at query time.
7. [src/Prediction/cli.py](../src/Prediction/cli.py) — How the interface presents results to users.
