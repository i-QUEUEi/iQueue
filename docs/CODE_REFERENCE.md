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
| [src/Evaluation/model_quality/model_evaluation.py](../src/Evaluation/model_quality/model_evaluation.py) | Tests how good each model is using 3 different methods |
| [src/Evaluation/model_quality/metrics.py](../src/Evaluation/model_quality/metrics.py) | Calculates the actual error scores (MAE, RMSE, R²) |
| [src/Evaluation/model_quality/splits.py](../src/Evaluation/model_quality/splits.py) | Splits data by date order (oldest → train, newest → test) |
| [src/Evaluation/outputs/plots.py](../src/Evaluation/outputs/plots.py) | Draws 4 charts to visualize model performance |
| [src/Evaluation/outputs/reporting.py](../src/Evaluation/outputs/reporting.py) | Writes a full human-readable report explaining everything |
| [src/Evaluation/outputs/samples.py](../src/Evaluation/outputs/samples.py) | Runs a few test predictions to make sure the model "makes sense" |
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

Since real LTO CDO queue records aren't always available, this file **simulates 44 weeks (~308 days) of realistic queue data**. It's not random noise — it's engineered with **12 layered realism factors** that combine to produce data statistically indistinguishable from a real government office queue.

> **Analogy:** It's like a movie prop department building a fake hospital room. It's not a real hospital, but everything looks and behaves like one so the actors (models) can practice realistically. Each factor below is a different prop — the beeping monitors, the sterile smell, the right lighting — all combining to create a convincing whole.

The output is saved to: `data/synthetic_lto_cdo_queue_90days.csv`

---

#### 🔧 Factor 0 — Reproducible Randomness ([Line 9](../data/Data_.py#L9))

```python
np.random.seed(42)
```

Every "random" number in the file is generated from a **fixed seed**. This means if you run the script 100 times, you get the **exact same dataset** every time.

**Why it matters:** Without this, every run would produce slightly different data → the model would train on different patterns each time → your results would never be reproducible. In research and production, reproducibility is non-negotiable.

> **Analogy:** It's like shuffling a deck of cards the exact same way every time. The cards *look* random, but you can always recreate the same shuffle.

---

#### 📋 Factor 1 — Hand-Crafted Daily Wait-Time Profiles ([Lines 19–26](../data/Data_.py#L19-L26))

```python
TRUE_PATTERNS = {
    'Monday':    [25, 55, 70, 60, 25, 50, 65, 55, 35],  # 8am to 4pm
    'Tuesday':   [12, 28, 32, 25, 15, 20, 30, 25, 15],
    'Wednesday': [ 9, 19, 25, 18,  9, 15, 22, 18, 12],
    'Thursday':  [11, 28, 32, 26, 16, 20, 30, 24, 15],
    'Friday':    [22, 50, 65, 55, 26, 48, 60, 51, 32],
    'Saturday':  [14, 28, 33, 25, 20, 24, 28, 25, 19]
}
```

Each list has **9 numbers** — one for each hour from 8 AM to 4 PM. These are the **baseline expected wait times in minutes** for that day+hour combination.

**What makes this realistic:**
- **Monday & Friday are 2–3× busier** than mid-week — matching real LTO patterns where people rush before/after the weekend.
- **Two peaks per day** — a morning surge (9–11 AM) and an afternoon surge (2–3 PM), with a lunch dip at noon. This "double-hump" shape is a well-documented pattern in service queues.
- **Wednesday is the calmest day** — the "valley" of the work week.
- **Saturday is moderate** — open but with reduced staffing.

> **Analogy:** Think of this as the "script" for how busy the office should be. Monday's script says "very busy morning, calm lunch, busy again in the afternoon." Wednesday's script says "chill all day."

---

#### 📅 Factor 2 — Philippine Holiday Calendar ([Lines 31–56](../data/Data_.py#L31-L56))

```python
def load_ph_holidays(calendar_path, year):
    # ... reads a CSV of Philippine holidays ...
    # Returns a set of datetime.date objects
    holidays.add(datetime(year, month, day).date())

holiday_dates = load_ph_holidays(HOLIDAY_CALENDAR_PATH, START_DATE.year)
```

The script reads a **real 2026 Philippine holiday calendar** file (`2026-calendar-with-holidays-portrait-sunday-start-en-ph.csv`) and extracts every holiday date.

**What makes this realistic:** Philippine holidays like Holy Week, Independence Day, and Christmas affect LTO offices differently than regular days. The system uses these actual dates — not made-up ones — so the model learns the *real* holiday distribution throughout the year.

---

#### 📈 Factor 3 — Long-Term Trend ([Line 64](../data/Data_.py#L64))

```python
trend_factor = 1.0 + 0.01 * np.sin(2 * np.pi * week / NUM_WEEKS) + 0.006 * (week / NUM_WEEKS)
```

This creates a **slow upward drift** in wait times over the 44-week period, combined with a **gentle wave**:

| Component | Effect |
|---|---|
| `0.006 * (week / NUM_WEEKS)` | Linear growth — wait times are ~0.6% longer by end of dataset |
| `0.01 * np.sin(...)` | A sine wave that makes the growth non-linear — some weeks are slightly higher/lower |

**Why it matters:** Real queue systems don't stay perfectly static. Population growth, policy changes, and seasonal demand cause gradual shifts. A model trained on perfectly flat data would miss these trends.

> **Analogy:** Like inflation — prices don't stay the same forever. There's a slow upward drift that's hard to notice day-by-day but adds up over months.

---

#### 🌦️ Factor 4 — Seasonal / Monthly Effect ([Lines 78–79](../data/Data_.py#L78-L79))

```python
month_angle = 2 * np.pi * (month - 1) / 12
seasonal_factor = 1.0 + 0.08 * np.sin(month_angle)
```

This creates an **8% seasonal swing** using a sine curve across the year:
- **Peak months** (around April–June): wait times are up to 8% longer
- **Low months** (around October–December): wait times are up to 8% shorter

**Why it matters:** LTO offices have seasonal demand — license renewal deadlines, registration cycles, and year-end rushes create predictable monthly variation.

> **Analogy:** Like how ice cream shops are busier in summer and quieter in winter. The "season" affects how many people show up.

---

#### 📆 Factor 5 — End-of-Month Rush ([Lines 80–82](../data/Data_.py#L80-L82))

```python
last_day = calendar.monthrange(day_date.year, month)[1]
is_end_of_month = 1 if day_date.day >= last_day - 2 else 0
eom_factor = 1.06 if is_end_of_month else 1.0
```

The **last 3 days of every month** get a **6% wait time increase**.

**Why it matters:** Government offices see end-of-month spikes because:
- Deadlines for license renewals and registrations often fall on month-end
- People procrastinate and rush to beat the deadline
- Paydays trigger more transactions

---

#### 🎌 Factor 6 — Holiday & Pre-Holiday Effects ([Lines 83–86](../data/Data_.py#L83-L86))

```python
is_holiday = 1 if day_date.date() in holiday_dates else 0
is_pre_holiday = 1 if (day_date + timedelta(days=1)).date() in holiday_dates else 0
holiday_factor = 0.75 if is_holiday else 1.0        # 25% FEWER people
pre_holiday_factor = 1.12 if is_pre_holiday else 1.0 # 12% MORE people
```

Two separate effects:

| Scenario | Factor | Why |
|---|---|---|
| **On a holiday** | 0.75× (25% reduction) | The office may be closed or running with skeleton staff — fewer people come |
| **Day before a holiday** | 1.12× (12% increase) | Everyone rushes to finish business before the office closes tomorrow |

**Why it matters:** The day *before* a holiday is often the busiest day of the week — people panic and crowd the office. The holiday itself is quieter. This asymmetric pattern is crucial for realistic predictions.

> **Analogy:** The supermarket is packed the day before a typhoon (pre-holiday rush), but empty during the typhoon itself (holiday quiet).

---

#### 🌊 Factor 7 — Multi-Frequency Day Waves ([Lines 89–90](../data/Data_.py#L89-L90))

```python
day_wave = 1.0 + 0.03 * np.sin(2 * np.pi * day_index / 9) + 0.02 * np.cos(2 * np.pi * day_index / 17)
baseline_load = 1.0 + 0.04 * np.sin(2 * np.pi * day_index / 11) + 0.03 * np.cos(2 * np.pi * day_index / 23)
```

These are **overlapping sine waves with different periods** (9, 11, 17, and 23 days). They create irregular, organic-looking variation that repeats at different intervals.

**Why it matters:** Real queue data isn't perfectly periodic. Some weeks are busier than others for no obvious reason — maybe a nearby school had an event, maybe it rained, maybe a viral social media post reminded people to renew their licenses. These waves simulate that **"unexplained but patterned" variation**.

> **Analogy:** Ocean waves. You don't get one clean wave — you get multiple overlapping waves of different sizes creating a complex but natural-looking pattern.

---

#### 🔄 Factor 8 — Carry-Over Load ([Lines 91–92](../data/Data_.py#L91-L92))

```python
day_load = 0.6 * prev_day_load + 0.4 * baseline_load
prev_day_load = day_load
```

This is an **exponential moving average** — today's "load" is 60% of yesterday's load + 40% new baseline. This creates **temporal autocorrelation** (busy days tend to be followed by busy days).

**Why it matters:** In real life, if Monday was slammed, Tuesday often starts with a backlog. People who couldn't be served Monday come back Tuesday. This carry-over effect is critical for the model to learn sequential dependencies.

> **Analogy:** Like momentum in a car. If you were going 80 km/h yesterday, you don't suddenly start at 0 today — there's inertia.

---

#### 🏢 Factor 9 — Staffing / Capacity Variation ([Line 93](../data/Data_.py#L93))

```python
capacity_factor = 1.03 if day_name in ['Monday', 'Friday'] else 0.98 if day_name in ['Tuesday', 'Wednesday', 'Thursday'] else 1.0
```

| Day type | Factor | Interpretation |
|---|---|---|
| Monday, Friday | 1.03 (+3%) | Busier days → longer waits even with similar staffing |
| Tue, Wed, Thu | 0.98 (−2%) | Mid-week calm → slightly shorter waits |
| Saturday | 1.00 | Neutral |

**Why it matters:** This simulates the effect of **staffing allocation** — real offices sometimes schedule fewer windows on days they expect to be quieter, and the per-person wait adjusts accordingly.

---

#### ⏰ Factor 10 — Hourly Micro-Oscillation ([Line 97](../data/Data_.py#L97))

```python
hour_wave = 1.0 + 0.025 * np.sin(2 * np.pi * (hour_idx + day_index) / 9)
```

A **small 2.5% oscillation** that shifts based on both the hour AND the day index. This means the same hour doesn't have the exact same multiplier every day.

**Why it matters:** Without this, every Monday at 9 AM would look nearly identical. In reality, there's always slight hour-to-hour variation that isn't explained by the day-level factors.

---

#### 🎲 Factor 11 — Controlled Noise Injection ([Lines 110–119](../data/Data_.py#L110-L119))

```python
for t in range(num_transactions):
    minute = np.random.randint(0, 60)
    arrival_time = day_date.replace(hour=hour, minute=minute, second=0)

    # Two layers of noise:
    wait_variation = np.random.uniform(0.85, 1.15)   # ±15% variation
    wait_time = base_wait * wait_variation
    noise = np.random.uniform(-0.08, 0.08)            # ±8% additional noise
    wait_time = wait_time * (1.0 + noise)
    wait_time = max(5, min(90, wait_time))             # Clamp to [5, 90] minutes
```

Each individual transaction gets **two layers of randomness**:

| Layer | Range | Purpose |
|---|---|---|
| `wait_variation` | ±15% | Simulates person-to-person differences (simple vs complex transactions) |
| `noise` | ±8% | Simulates unpredictable system delays (printer jams, staff breaks) |
| `clamp [5, 90]` | Hard limits | No one waits less than 5 min or more than 90 min — prevents impossible values |

**Why it matters:** If every Monday 9 AM had the *exact same* wait time, the model would just memorize a lookup table. The noise forces the model to learn **patterns** rather than **specific numbers**. But the noise is bounded — it can't overwhelm the signal.

> **Analogy:** Two students who studied the same amount won't score the exact same mark. There's natural variation — but a student who studied 20 hours will still consistently score higher than one who studied 2 hours.

---

#### 📊 Factor 12 — Queue Length ↔ Wait Time Correlation ([Lines 122–132](../data/Data_.py#L122-L132))

```python
# Queue length is correlated with wait time
if wait_time > 45:
    queue_length = np.random.randint(25, 45)
elif wait_time > 25:
    queue_length = np.random.randint(12, 28)
else:
    queue_length = np.random.randint(2, 12)

# Nonlinear congestion: longer queues grow waits faster
if queue_length > 18:
    wait_time *= 1.0 + (queue_length - 18) / 90
    wait_time = max(5, min(90, wait_time))
```

This creates a **bidirectional relationship**:

1. **Wait → Queue:** High wait times produce high queue lengths (people accumulate).
2. **Queue → Wait (nonlinear):** If the queue exceeds 18 people, wait times increase *even further* — each additional person above 18 adds progressively more delay.

| Wait Time Range | Queue Length Range | Congestion Level |
|---|---|---|
| < 25 min | 2–12 people | 🟢 Low |
| 25–45 min | 12–28 people | 🟡 Moderate |
| > 45 min | 25–45 people | 🔴 High |

**Why it matters:** In real queues, congestion is **nonlinear** — going from 10 to 15 people might add 5 minutes, but going from 30 to 35 might add 15 minutes because the system is overwhelmed. This teaches the model that queue length isn't just a number — it has an *accelerating* effect.

> **Analogy:** Traffic on a highway. At 50% capacity, everything flows smoothly. At 80% capacity, things slow down. At 95% capacity, everything gridlocks — a small increase causes a massive slowdown.

---

#### 🔧 Factor 13 — Service Time Correlation ([Lines 135–140](../data/Data_.py#L135-L140))

```python
if wait_time > 45:
    service_time = np.random.uniform(45, 75)   # Long wait → long service
elif wait_time > 25:
    service_time = np.random.uniform(30, 50)   # Medium wait → medium service
else:
    service_time = np.random.uniform(15, 35)   # Short wait → short service
```

Service time (how long you spend at the window) is **correlated with congestion** — when the office is busy, transactions take longer because:
- Staff are stressed and slower
- Complex cases pile up during busy hours
- Systems run slower under load

---

#### 📉 Factor 14 — Transaction Volume Variation ([Lines 101–105](../data/Data_.py#L101-L105))

```python
if is_holiday:
    num_transactions = np.random.randint(4, 9)    # Fewer people on holidays
else:
    num_transactions = np.random.randint(8, 15)   # Normal day
```

Each hour generates a **different number of transaction records**. Holidays produce fewer rows; normal days produce more.

**Why it matters:** This means the dataset is **not uniformly distributed** — some hours/days have more data points than others, just like real data. The model must handle this imbalance gracefully.

---

#### 🔗 Factor 15 — Lag Features ([Lines 165–184](../data/Data_.py#L165-L184))

```python
# What happened in the PREVIOUS transaction?
df['queue_length_lag1'] = df.groupby('date')['queue_length_at_arrival'].shift(1)
df['waiting_time_lag1'] = df.groupby('date')['waiting_time_min'].shift(1)

# First transaction of the day gets sensible defaults:
if day_name in ['Monday', 'Friday']:
    df.loc[first_idx, 'queue_length_lag1'] = 8     # Busy day → start with 8 in queue
    df.loc[first_idx, 'waiting_time_lag1'] = 25    # Busy day → expect 25 min wait
else:
    df.loc[first_idx, 'queue_length_lag1'] = 3     # Calm day → start with 3 in queue
    df.loc[first_idx, 'waiting_time_lag1'] = 10    # Calm day → expect 10 min wait
```

Lag features give the model **"memory"** — it can see what the queue looked like one transaction ago. This is critical because queues have **momentum**: if the last 3 people waited 40+ minutes, the next person probably will too.

The `groupby('date')` ensures that the lag is reset at the start of each day — yesterday's last transaction doesn't leak into today's first.

---

#### 🏷️ Factor 16 — Binary Feature Flags ([Lines 107–108](../data/Data_.py#L107-L108), [189](../data/Data_.py#L189))

```python
is_peak_day = 1 if day_name in ['Monday', 'Friday'] else 0
is_peak_hour = 1 if hour in [9, 10, 11, 14, 15] else 0
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
```

These are **pre-computed indicator features** that make it easier for the model to spot important patterns without having to "figure out" that day_of_week=0 means Monday:

| Flag | When it's 1 | Why the model needs it |
|---|---|---|
| `is_peak_day` | Monday or Friday | Quick shortcut for "expect high volume" |
| `is_peak_hour` | 9, 10, 11 AM or 2, 3 PM | Quick shortcut for "expect long waits" |
| `is_weekend` | Saturday | Different staffing, shorter hours |

---

#### 🧮 Factor 17 — Cyclical Month Encoding ([Lines 148–150](../data/Data_.py#L148-L150))

```python
'month_sin': round(float(np.sin(month_angle)), 6),
'month_cos': round(float(np.cos(month_angle)), 6),
```

Instead of storing the month as a plain number (1–12), it's encoded as **sine and cosine values**. This is because months are **cyclical** — December (12) is actually close to January (1), but the number 12 is far from 1. Sine/cosine encoding preserves this circular relationship.

| Month | month_sin | month_cos |
|---|---|---|
| January (1) | 0.000 | 1.000 |
| April (4) | 1.000 | 0.000 |
| July (7) | 0.000 | −1.000 |
| December (12) | −0.500 | 0.866 |

> **Analogy:** Imagine a clock. The number 12 is right next to 1 on the clock face, even though the numbers look far apart. Sine/cosine encoding is like giving the model a clock instead of a number line.

---

#### 📊 Summary: How All 17 Factors Layer Together

Each transaction's wait time is built by **multiplying the base pattern through a cascade of modifiers**:

```
final_wait = TRUE_PATTERN[day][hour]
           × seasonal_factor        (monthly cycle)
           × eom_factor             (end-of-month rush)
           × holiday_factor         (holiday quiet)
           × pre_holiday_factor     (pre-holiday rush)
           × trend_factor           (long-term growth)
           × day_wave               (multi-frequency variation)
           × hour_wave              (hourly micro-oscillation)
           × day_load               (carry-over from yesterday)
           × capacity_factor        (staffing adjustment)
           × wait_variation         (person-to-person noise)
           × (1 + noise)            (system noise)
           × congestion_boost       (nonlinear queue penalty)
           → clamped to [5, 90] minutes
```

This multiplicative design means the factors **interact** — a Monday + end-of-month + pre-holiday can stack to produce extreme but realistic peaks, while a Wednesday + mid-month + holiday produces the calmest possible scenario. The model must learn these interactions, not just individual effects.

> **Final analogy:** Think of it like cooking a sauce. Each factor is a spice. Salt alone is too simple. But salt + pepper + garlic + lime + chili + a slow simmer creates something complex and realistic. No single factor makes the data "real" — it's all 17 working together.

---

### Step 3 — Clean and prepare the data

**Files:** [loader.py](../src/Preprocessing/loader.py), [features.py](../src/Preprocessing/features.py), [calendar.py](../src/Preprocessing/calendar.py), [preprocess.py](../src/Preprocessing/preprocess.py)

Raw data is messy. This step cleans it and adds columns that make the model smarter. There are **4 files** that work together, each with a specific job.

> **Analogy:** Think of this like a chef prepping ingredients. The raw chicken (data) needs to be washed, cut, and seasoned (cleaned + feature-engineered) before it goes in the oven (model). Each file below is a different prep station in the kitchen.

---

#### 📥 Sub-step 3a — Load and clean the CSV ([loader.py](../src/Preprocessing/loader.py))

The `load_data()` function ([Lines 8–43](../src/Preprocessing/loader.py#L8-L43)) does **three things** in sequence:

**1. Read the CSV and parse dates** ([Lines 9–10](../src/Preprocessing/loader.py#L9-L10))
```python
df = pd.read_csv(path)
df["date"] = pd.to_datetime(df["date"])
```
Converts the `date` column from a plain string (`"2026-01-05"`) into a proper datetime object so we can extract day, month, and week programmatically.

**2. Engineer new columns** ([Lines 12–28](../src/Preprocessing/loader.py#L12-L28))

```python
df["month"] = df["date"].dt.month
month_angle = 2 * np.pi * (df["month"] - 1) / 12
df["month_sin"] = np.sin(month_angle)
df["month_cos"] = np.cos(month_angle)
df["is_end_of_month"] = (df["date"].dt.day >= (df["date"].dt.days_in_month - 2)).astype(int)
```

| New column | How it's computed | Why the model needs it |
|---|---|---|
| `month` | Extracted from date | Seasonal patterns (summer vs. year-end) |
| `month_sin`, `month_cos` | Sine/cosine of month angle | Cyclical encoding — December (12) wraps to January (1) |
| `is_end_of_month` | 1 if within last 3 days of month | End-of-month deadline rushes |
| `is_holiday` | Checked against PH holiday calendar ([L18–26](../src/Preprocessing/loader.py#L18-L26)) | Holidays reduce queue traffic |
| `is_pre_holiday` | Checked if *tomorrow* is a holiday ([L21–23](../src/Preprocessing/loader.py#L21-L23)) | Day before a holiday = rush |
| `week_of_month` | `(day - 1) // 7 + 1` ([L28](../src/Preprocessing/loader.py#L28)) | Week 1 vs week 4 have different patterns |

**Why `is_pre_holiday` matters:** The code checks `d + pd.Timedelta(days=1)` — it looks at *tomorrow's* date. If tomorrow is a holiday, today gets flagged because people rush to finish business before the office closes.

**3. Remove bad rows** ([Lines 30–32](../src/Preprocessing/loader.py#L30-L32))

```python
df = df[df["waiting_time_min"] >= 0]       # No negative wait times
df = df[df["queue_length_at_arrival"] >= 0] # No negative queue lengths
df = df.dropna()                            # No rows with missing values
```

**Why this matters:** A negative wait time is physically impossible — it would mean someone was served *before* they arrived. These rows are data corruption and must be removed or the model learns nonsense. `dropna()` removes any row with a blank cell — garbage in, garbage out.

> **Analogy:** You wouldn't put a rotten tomato in your sauce. These three lines are the quality inspection that tosses out any bad ingredients.

---

#### 📋 Sub-step 3b — Define the feature list ([features.py — Lines 14–31](../src/Preprocessing/features.py#L14-L31))

```python
FEATURES = [
    "hour",                    # What time of day (8–16)
    "day_of_week",             # Monday=0 through Saturday=5
    "week_of_month",           # Week 1, 2, 3, or 4
    "month",                   # January=1 through December=12
    "month_sin", "month_cos",  # Cyclical month encoding
    "is_end_of_month",         # Last 3 days of month flag
    "is_holiday",              # Philippine holiday flag
    "is_pre_holiday",          # Day before a holiday flag
    "is_peak_day",             # Monday or Friday flag
    "queue_length_at_arrival", # How many people are already in line
    "service_time_min",        # How long the current transaction takes
    "is_weekend",              # Saturday flag
    "is_peak_hour",            # 9–11am or 2–3pm flag
    "queue_length_lag1",       # Queue length of previous transaction
    "waiting_time_lag1",       # Wait time of previous transaction
]
```

This is the **exact list of 16 columns** the model is allowed to see during training. The target column (`waiting_time_min`) is deliberately excluded — that's what the model is trying to *predict*, so it can't be an input.

The `get_features()` function ([Lines 34–48](../src/Preprocessing/features.py#L34-L48)) extracts these columns and returns `X` (inputs) and `y` (target):

```python
def get_features(df):
    X = df[FEATURES]              # 16 input columns
    y = df["waiting_time_min"]    # Target: what we're predicting
    return X, y, FEATURES
```

**Why this list matters:** Every feature was chosen for a reason. Removing `queue_length_lag1` would strip the model of its "memory." Removing `is_peak_day` would force the model to figure out on its own that Monday and Friday are special. The feature list is the model's **vocabulary** — if a word isn't in the vocabulary, the model can't use it.

---

#### 📅 Sub-step 3c — Parse the holiday calendar ([calendar.py](../src/Preprocessing/calendar.py))

```python
def load_ph_holiday_month_days(calendar_path: Path):
    holiday_md = set()
    text = calendar_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        match = re.search(r"\b([A-Za-z]{3})\s+(\d{1,2})\s*:\s*", line)
        if not match:
            continue
        month_name = match.group(1).title()   # "Jan" → "Jan"
        day = int(match.group(2))             # "1" → 1
        month = MONTH_MAP.get(month_name)     # "Jan" → 1
        if month:
            holiday_md.add((month, day))       # Store as (1, 1) for Jan 1
    return holiday_md
```

**How it works line by line** ([Lines 20–35](../src/Preprocessing/calendar.py#L20-L35)):

1. Opens the holiday CSV as plain text
2. Uses a **regex pattern** `\b([A-Za-z]{3})\s+(\d{1,2})\s*:\s*` to find lines like `"Jan 1 : New Year's Day"`
3. Extracts the month abbreviation (`Jan`) and day number (`1`)
4. Converts to `(month, day)` tuple using `MONTH_MAP` ([Lines 4–17](../src/Preprocessing/calendar.py#L4-L17))
5. Stores in a `set` for **O(1) lookup** — checking "is this a holiday?" takes the same time whether there are 10 or 1,000 holidays

> **Analogy:** It's like building a quick-reference index card with all the holiday dates. Instead of reading the entire calendar every time someone asks "is March 28 a holiday?", you just check the index card instantly.

---

#### 🔗 Sub-step 3d — The convenience hub ([preprocess.py](../src/Preprocessing/preprocess.py))

```python
from .calendar import load_ph_holiday_month_days
from .features import FEATURES, build_feature_dataframe, get_features
from .loader import load_data
```

This file ([Lines 1–11](../src/Preprocessing/preprocess.py#L1-L11)) doesn't contain any logic — it **re-exports** everything from the other 3 files so the rest of the codebase can import from one place:

```python
# Instead of this (3 separate imports):
from Preprocessing.loader import load_data
from Preprocessing.features import get_features
from Preprocessing.calendar import load_ph_holiday_month_days

# You can do this (1 import):
from Preprocessing.preprocess import load_data, get_features
```

**Why this design:** It keeps the code organized. If a function moves from `loader.py` to `features.py`, only `preprocess.py` needs to change — not every file that imports it.

---

### Step 4 — Test all models

**Files:** [train_model.py](../src/model_implementation/train_model.py), [model_zoo/](../src/model_implementation/model_zoo/__init__.py), [evaluation.py](../src/Evaluation/model_quality/model_evaluation.py)

Three models compete against each other. Each is defined in its own file, assembled into a catalog, then trained and evaluated in a loop.

---

#### 🏗️ The Model Catalog ([model_zoo/\_\_init\_\_.py — Lines 6–11](../src/model_implementation/model_zoo/__init__.py#L6-L11))

```python
def build_model_catalog(random_state):
    return {
        "LinearRegression": build_linear_regression(),
        "RandomForest": build_random_forest(random_state),
        "GradientBoosting": build_gradient_boosting(random_state),
    }
```

This function returns a **dictionary** of model name → model object. The training loop ([train_model.py — Lines 72–140](../src/model_implementation/train_model.py#L72-L140)) iterates over this dictionary, training and evaluating each model identically:

```python
models = build_model_catalog(RANDOM_STATE)
for name, model in models.items():
    result = evaluate_model(name, model, X_train, X_test, y_train, y_test, ...)
```

**Why a catalog:** Adding a 4th model (e.g., XGBoost) only requires creating a new builder file and adding one line to this dictionary — no changes to the training loop.

---

#### 📐 Model #1 — Linear Regression ([linear_regression.py — Lines 4–5](../src/model_implementation/model_zoo/linear_regression.py#L4-L5))

```python
def build_linear_regression():
    return LinearRegression()
```

**How it works:** Fits a formula `ŷ = β₀ + β₁x₁ + β₂x₂ + ... + β₁₆x₁₆` where each `β` is a weight that says "how much does this feature affect wait time?"

| Property | Value |
|---|---|
| Parameters | None — uses sklearn defaults |
| Training speed | ⚡ Fastest (milliseconds) |
| Formula | One weight per feature, added together |
| Strengths | Simple, interpretable, fast |
| Weaknesses | Can't capture non-linear interactions (e.g., "Monday + 10am = extra bad") |

During training, the pipeline prints the top coefficients ([train_model.py — Lines 92–99](../src/model_implementation/train_model.py#L92-L99)):
```python
print(f"   │   Intercept (β₀)  : {fitted.intercept_:.4f}")
coef_pairs = sorted(zip(features, fitted.coef_), key=lambda x: abs(x[1]), reverse=True)
for feat, coef in coef_pairs[:5]:
    direction = "↑" if coef > 0 else "↓"
    print(f"   │     {feat:<30} β = {coef:+.4f}  {direction}")
```

> **Analogy:** Drawing a straight line through all the data points. Simple, fast, and easy to explain. It works if the relationship is roughly linear (e.g., "more people in queue = longer wait"). It won't capture complex patterns though.

---

#### 🌲 Model #2 — Random Forest ([random_forest.py — Lines 4–15](../src/model_implementation/model_zoo/random_forest.py#L4-L15))

```python
def build_random_forest(random_state, params=None):
    default_params = {
        "n_estimators": 500,        # Build 500 decision trees
        "max_depth": 15,            # Each tree can be at most 15 levels deep
        "min_samples_split": 5,     # Need at least 5 samples to split a node
        "min_samples_leaf": 2,      # Each leaf must have at least 2 samples
        "max_features": "sqrt",     # Each tree only sees √16 ≈ 4 random features
        "random_state": random_state,
    }
    return RandomForestRegressor(**default_params)
```

**How it works:** Builds 500 separate decision trees, each trained on a **random subset** of the data and features. The final prediction is the **average** of all 500 trees' predictions.

| Parameter | Value | Why this value |
|---|---|---|
| `n_estimators` | 500 | More trees = more stable predictions, diminishing returns after ~300 |
| `max_depth` | 15 | Deep enough to learn "Monday + 10am + end-of-month" combos, shallow enough to avoid memorization |
| `min_samples_split` | 5 | Prevents splits on tiny groups that might be noise |
| `min_samples_leaf` | 2 | Every leaf must represent at least 2 real data points |
| `max_features` | `"sqrt"` | Each tree sees only ~4 of 16 features → forces diversity among trees |

During training, tree depth statistics are printed ([train_model.py — Lines 100–109](../src/model_implementation/train_model.py#L100-L109)):
```python
depths = [t.get_depth() for t in fitted.estimators_]
print(f"   │   Actual depths: min={min(depths)}, avg={sum(depths)/len(depths):.1f}, max={max(depths)}")
```

> **Analogy:** Ask 500 different people for their opinion, then take the average. Each "person" is a decision tree trained on slightly different data. The combined answer is more reliable than any single tree.

Handles non-linear patterns naturally (e.g., "Monday mornings are bad BUT only during the last week of the month").

---

#### 🚀 Model #3 — Gradient Boosting ([gradient_boosting.py — Lines 4–14](../src/model_implementation/model_zoo/gradient_boosting.py#L4-L14))

```python
def build_gradient_boosting(random_state, params=None):
    default_params = {
        "n_estimators": 250,       # Build 250 trees sequentially
        "learning_rate": 0.05,     # Each tree corrects only 5% of the remaining error
        "max_depth": 3,            # Shallow trees — each captures a simple pattern
        "subsample": 0.9,          # Each tree only sees 90% of the training data
        "random_state": random_state,
    }
    return GradientBoostingRegressor(**default_params)
```

**How it works:** Builds trees **one at a time**, where each new tree focuses on correcting the **mistakes** of all previous trees combined. The formula is `ŷ = F₀ + η·h₁(x) + η·h₂(x) + ... + η·h₂₅₀(x)` where `η` (eta) is the learning rate.

| Parameter | Value | Why this value |
|---|---|---|
| `n_estimators` | 250 | 250 sequential correction rounds |
| `learning_rate` | 0.05 | Small steps prevent overcorrection — each tree only fixes 5% of the error |
| `max_depth` | 3 | Deliberately shallow — each tree learns one simple rule, complexity comes from combining 250 of them |
| `subsample` | 0.9 | Random 10% holdout per tree adds regularization (noise resistance) |

During training, the formula and stage count are printed ([train_model.py — Lines 110–118](../src/model_implementation/train_model.py#L110-L118)):
```python
print(f"   │   Trees (stages): {fitted.n_estimators_}  |  Learning rate (η): {fitted.learning_rate}")
print(f"   │   Max depth per tree: {fitted.max_depth}  |  Subsample: {fitted.subsample}")
```

> **Analogy:** Imagine a student who keeps reviewing only the questions they got wrong. Each new tree in gradient boosting focuses on fixing the previous tree's mistakes. It's slower to train but often more accurate.

---

#### 🏆 The Selection Logic ([train_model.py — Lines 139–140](../src/model_implementation/train_model.py#L139-L140))

```python
if selected_result is None or result["robust_mae"] < selected_result["robust_mae"]:
    selected_result = result
```

After all 3 models are evaluated, the one with the **lowest `robust_mae`** wins. The winning model is saved to disk ([Line 172](../src/model_implementation/train_model.py#L172)):

```python
joblib.dump(selected_result["model"], MODEL_PATH)  # → models/queue_model.pkl
```

`joblib.dump()` serializes the entire trained model object to a `.pkl` file — including all 500 trees (for Random Forest) or all 250 stages (for Gradient Boosting). This file is what the prediction system loads later.

---

### Step 5 — Evaluate each model (3 different ways)

**File:** [evaluation.py](../src/Evaluation/model_quality/model_evaluation.py)

Just checking accuracy on one test set is not enough — a model might get lucky. So each model is tested **3 ways** inside the `evaluate_model()` function ([Lines 33–134](../src/Evaluation/model_quality/model_evaluation.py#L33-L134)).

---

#### 🎲 Way 1 — Random Split ([model_evaluation.py — Lines 45–49](../src/Evaluation/model_quality/model_evaluation.py#L45-L49))

```python
fitted_model = clone(model)          # Fresh copy of the model
fitted_model.fit(X_train, y_train)   # Train on 80%
train_pred = fitted_model.predict(X_train)  # Predict on training data
test_pred = fitted_model.predict(X_test)    # Predict on unseen 20%
```

The actual 80/20 split is done earlier in [train_model.py — Line 61](../src/model_implementation/train_model.py#L61):
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**How `clone(model)` works:** `sklearn.base.clone()` creates a brand-new copy of the model with the same parameters but **no training data**. This ensures each evaluation method starts with a completely fresh model — no leftover learning from a previous test.

> **Analogy:** Shuffle a deck of cards and deal 20% out as your "quiz cards." The model hasn't seen them before. This tests general accuracy, but has a weakness — future data might look different from randomly shuffled past data.

---

#### 📅 Way 2 — Chronological Split ([model_evaluation.py — Lines 51–53](../src/Evaluation/model_quality/model_evaluation.py#L51-L53), using [splits.py](../src/Evaluation/model_quality/splits.py))

```python
chrono_model = clone(model)
chrono_model.fit(chrono_train[0], chrono_train[1])    # Train on oldest 80% of dates
chrono_pred = chrono_model.predict(chrono_test[0])    # Predict on newest 20%
```

The split logic in [splits.py — Lines 4–23](../src/Evaluation/model_quality/splits.py#L4-L23):

```python
def chronological_split(df, features):
    df_time = df.sort_values("date").copy()                    # Sort oldest → newest
    unique_dates = np.sort(normalized_dates.unique())          # Get unique dates
    split_idx = int(len(unique_dates) * 0.8)                   # 80% cutoff
    time_train_dates = unique_dates[:split_idx]                # Jan–Aug for training
    time_test_dates = unique_dates[split_idx:]                 # Sep–Nov for testing
```

**Why this is different from random split:** Random split can "leak" future information. If the model trains on a Friday in October and tests on the preceding Wednesday, it's technically seeing the future. Chronological split guarantees the model has **never seen any data from the test dates** — just like in production, where you can only predict the future, not look it up.

**Key detail:** The split is by **unique dates**, not rows. A Monday with 120 rows and a Wednesday with 80 rows each count as one date. This prevents busy days from dominating the split.

> **Analogy:** Study using notes from January–August, then take the September–November exam. This tests whether the model can handle *future* dates, not just random ones from the middle of the dataset.

---

#### 🔄 Way 3 — 5-Fold Cross-Validation ([model_evaluation.py — Lines 55–66](../src/Evaluation/model_quality/model_evaluation.py#L55-L66))

```python
cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
cv_results = cross_validate(
    clone(model),
    X_train, y_train,
    cv=cv,
    scoring={
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    },
)
```

**How it works step by step:**
1. The training data is split into **5 equal-size chunks** (folds)
2. **Round 1:** Train on folds 2-3-4-5, test on fold 1
3. **Round 2:** Train on folds 1-3-4-5, test on fold 2
4. **Round 3–5:** Same rotation pattern
5. Average the 5 test scores → one stable score

**Why `shuffle=True`:** Without shuffling, fold 1 would always be the earliest data and fold 5 the latest — making this just another chronological split. Shuffling mixes the data so each fold contains a random sample from all time periods.

**Why scores are negative:** Sklearn's convention uses negative values for error metrics (so higher = better internally). The code flips them back ([Lines 71–73](../src/Evaluation/model_quality/model_evaluation.py#L71-L73)):
```python
cv_mae = float(-cv_results["test_mae"].mean())     # Flip sign back
cv_rmse = float(-cv_results["test_rmse"].mean())
cv_r2 = float(cv_results["test_r2"].mean())          # R² is already positive
```

> **Analogy:** Take 5 different practice exams from different tutors. Your average score across all 5 is more reliable than any single exam score.

---

#### 📊 Combining all 3 into Robust MAE ([Line 75](../src/Evaluation/model_quality/model_evaluation.py#L75))

```python
robust_mae = float(np.mean([test_metrics["mae"], chrono_metrics["mae"], cv_mae]))
```

This averages the MAE from all 3 methods into a single number. **Why average?** Each method tests a different weakness:

| Method | What it catches |
|---|---|
| Random split | Overall prediction quality |
| Chronological split | Can the model generalize to *future* dates? |
| 5-Fold CV | Is the result stable or did we get lucky with one split? |

A model that scores well on all 3 is genuinely good — not just lucky.

---

#### 🔍 Additional diagnostics ([Lines 77–107](../src/Evaluation/model_quality/model_evaluation.py#L77-L107))

After the 3-way evaluation, the function also computes:

**Percentile errors** ([Lines 77–80](../src/Evaluation/model_quality/model_evaluation.py#L77-L80)):
```python
abs_errors = np.abs(y_test.to_numpy() - test_pred)
p90_abs_error = float(np.percentile(abs_errors, 90))   # 90% of predictions are within this
p95_abs_error = float(np.percentile(abs_errors, 95))   # 95% of predictions are within this
max_abs_error = float(np.max(abs_errors))               # Worst single prediction
```

**Segment error breakdown** ([Lines 96–105](../src/Evaluation/model_quality/model_evaluation.py#L96-L105)):
- Error by **day of week** — is the model worse on Mondays vs Wednesdays?
- Error by **hour** — is the model worse at 10 AM vs 3 PM?
- Error for **peak vs non-peak** days and hours

**Feature importance** ([Lines 10–30](../src/Evaluation/model_quality/model_evaluation.py#L10-L30)):
```python
def get_feature_importance(model, features, X_reference, y_reference, random_state):
    if hasattr(model, "feature_importances_"):       # Random Forest / Gradient Boosting
        values = model.feature_importances_
    elif hasattr(model, "coef_"):                    # Linear Regression
        values = np.abs(model.coef_)
    else:                                            # Fallback: permutation importance
        importance = permutation_importance(model, X_reference, y_reference, ...)
```

This tells you **which features the model relies on most**. If `queue_length_at_arrival` has importance 0.35 and `is_end_of_month` has 0.01, the model cares 35× more about queue length than end-of-month status.

### Step 6 — Evaluate the data itself

**File:** [model_evaluation.py — Lines 137–155](../src/Evaluation/data_quality/data_evaluation.py-L155)

Before training even begins, `evaluate_data_quality()` audits the **raw CSV** to catch problems early. It's called at [train_model.py — Lines 48–52](../src/model_implementation/train_model.py#L48-L52):

```python
raw_df = pd.read_csv(DATA_PATH)
summary = evaluate_data_quality(raw_df)
```

The function computes a **dictionary of 14 statistics** ([Lines 141–155](../src/Evaluation/data_quality/data_evaluation.py#L141-L155)):

```python
def evaluate_data_quality(raw_df):
    target = raw_df["waiting_time_min"]
    queue = raw_df["queue_length_at_arrival"]
    return {
        "rows": int(len(raw_df)),                          # Total row count
        "columns": int(raw_df.shape[1]),                   # Total column count
        "duplicate_rows": int(raw_df.duplicated().sum()),  # Exact duplicate rows
        "missing_cells": int(raw_df.isna().sum().sum()),   # Blank cells across all columns
        "negative_waiting_rows": int((target < 0).sum()),  # Impossible negative waits
        "negative_queue_rows": int((queue < 0).sum()),     # Impossible negative queues
        "target_mean": float(target.mean()),               # Average wait time
        "target_median": float(target.median()),           # Middle-of-the-pack wait
        "target_std": float(target.std()),                 # How spread out the waits are
        "target_min": float(target.min()),                 # Shortest wait ever
        "target_p10": float(target.quantile(0.10)),        # 10th percentile
        "target_p90": float(target.quantile(0.90)),        # 90th percentile
        "target_max": float(target.max()),                 # Longest wait ever
    }
```

**Why each metric matters:**

| Metric | What you're checking | Red flag if... |
|---|---|---|
| `rows` | Is there enough data to train on? | Under ~1,000 rows → model may not learn well |
| `duplicate_rows` | Was data accidentally copied? | More than ~1% duplicates |
| `missing_cells` | Incomplete records | Any missing cells in key columns |
| `negative_waiting_rows` | Physically impossible values | Any count > 0 means data corruption |
| `target_std` | Variation in wait times | Near 0 means all waits are the same → nothing for the model to learn |
| `target_p10` / `target_p90` | Typical range of waits | If P10 ≈ P90, data has no signal |

This summary is printed during training and later saved to `outputs/metrics.txt` by [reporting.py — Lines 11–22](../src/Evaluation/outputs/reporting.py#L11-L22).

> **Analogy:** Before cooking a meal, a chef checks the ingredients — are they fresh? Any mold? Enough quantity? This step is the data equivalent of that inspection.

---

### Step 7 — Generate charts

**File:** [plots.py](../src/Evaluation/outputs/plots.py)

Four charts are saved to `outputs/plots/`, triggered from [train_model.py — Lines 176–183](../src/model_implementation/train_model.py#L176-L183):

```python
plot_target_distribution(df, PLOTS_DIR)
plot_day_hour_heatmap(df, PLOTS_DIR)
plot_model_comparison(results_df, PLOTS_DIR)
plot_actual_vs_predicted(y_test.to_numpy(), selected_result["test_pred"], selected_model, PLOTS_DIR)
```

**Why `matplotlib.use("Agg")`** ([Line 3](../src/Evaluation/outputs/plots.py#L3)): This sets matplotlib to "non-interactive" mode — it generates image files without trying to open a GUI window. Essential for servers and automated pipelines where no screen is available.

---

#### 📊 Chart 1 — Waiting Time Distribution ([Lines 8–16](../src/Evaluation/outputs/plots.py#L8-L16))

```python
def plot_target_distribution(df, plots_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["waiting_time_min"], bins=24, color="#1f77b4", alpha=0.85, edgecolor="white")
    ax.set_title("Waiting Time Distribution")
```

| Detail | Value | Why |
|---|---|---|
| Chart type | Histogram | Shows frequency distribution — how many transactions fall in each wait-time range |
| Bins | 24 | Divides the range into 24 buckets → ~3.5 min per bin |
| Saved as | `target_distribution.png` | |

**How to read it:** If most bars are on the left (short waits) with a tail to the right, the data is right-skewed — which is realistic for queues. A flat distribution would suggest the data is uniformly random (unrealistic).

---

#### 🌡️ Chart 2 — Day × Hour Heatmap ([Lines 19–37](../src/Evaluation/outputs/plots.py#L19-L37))

```python
def plot_day_hour_heatmap(df, plots_dir):
    heatmap = df.pivot_table(
        index="day_name", columns="hour",
        values="waiting_time_min", aggfunc="mean"
    ).reindex(day_order)
    im = ax.imshow(heatmap.values, aspect="auto", cmap="YlOrRd")
```

| Detail | Value | Why |
|---|---|---|
| Chart type | Heatmap (color matrix) | Shows 2D patterns — each cell is a day+hour combination |
| Color map | `YlOrRd` (Yellow → Orange → Red) | Yellow = short waits, red = long waits |
| Aggregation | Mean per cell | Averages across all weeks for that day+hour |
| Saved as | `day_hour_heatmap.png` | |

**How to read it:** Look for **hot spots** (red cells). You should see:
- Monday and Friday rows are redder overall (busier days)
- Columns 9–11 and 14–15 are redder (peak hours)
- The pattern forms a "double hump" per row (morning peak + afternoon peak)

---

#### 📊 Chart 3 — Model Comparison Bar Chart ([Lines 40–56](../src/Evaluation/outputs/plots.py#L40-L56))

```python
def plot_model_comparison(results_df, plots_dir):
    ax.bar(x - width, results_df["test_mae"], width=width, label="Random split MAE", color="#4c78a8")
    ax.bar(x, results_df["chrono_test_mae"], width=width, label="Chronological MAE", color="#f58518")
    ax.bar(x + width, results_df["robust_mae"], width=width, label="Robust MAE", color="#54a24b")
```

| Detail | Value | Why |
|---|---|---|
| Chart type | Grouped bar chart | 3 bars per model, side by side |
| Bars per model | Random split MAE, Chronological MAE, Robust MAE | Shows consistency across evaluation methods |
| Saved as | `model_comparison.png` | |

**How to read it:** The model with the **shortest green bar** (Robust MAE) wins. If a model has a short blue bar but a tall orange bar, it performs well on random data but poorly on future prediction — a red flag.

---

#### 🎯 Chart 4 — Actual vs Predicted Scatter ([Lines 59–69](../src/Evaluation/outputs/plots.py#L59-L69))

```python
def plot_actual_vs_predicted(y_true, y_pred, model_name, plots_dir):
    ax.scatter(y_true, y_pred, alpha=0.7, color="#1f77b4", edgecolor="white", linewidth=0.4)
    ax.plot(bounds, bounds, linestyle="--", color="#d62728", linewidth=2)  # Perfect prediction line
```

| Detail | Value | Why |
|---|---|---|
| Chart type | Scatter plot | Each dot is one prediction vs. its true value |
| X-axis | Actual wait time | What really happened |
| Y-axis | Predicted wait time | What the model guessed |
| Red dashed line | y = x (perfect predictions) | If all dots sit on this line, the model is perfect |
| Saved as | `actual_vs_predicted.png` | |

**How to read it:**
- **Dots clustered tightly around the red line** → model is accurate
- **Dots forming a cloud far from the line** → model is unreliable
- **Dots above the line** → model over-predicts (says wait is longer than reality)
- **Dots below the line** → model under-predicts (says wait is shorter)

> **Analogy:** It's like a teacher grading themselves. They write down what they *think* your score will be (y-axis), then compare to your *actual* score (x-axis). The closer to the diagonal, the better the teacher knows the class.

---

### Step 8 — Write the report

**File:** [reporting.py](../src/Evaluation/outputs/reporting.py)

The `write_report()` function ([Lines 4–78](../src/Evaluation/outputs/reporting.py#L4-L78)) generates a comprehensive human-readable report at `outputs/metrics.txt`. It's called from [train_model.py — Line 186](../src/model_implementation/train_model.py#L186):

```python
write_report(summary, results_df, selected_result, baseline_metrics, time_train_dates, time_test_dates, OUTPUTS_DIR)
```

The report has **9 sections**, each documenting a different aspect of the training run:

| Section | Lines | What it contains | Why it matters |
|---|---|---|---|
| **DATA EVALUATION** | [L11–22](../src/Evaluation/outputs/reporting.py#L11-L22) | Row count, duplicates, missing values, wait time stats (mean, median, std, min, P10, P90, max) | Proves the data is clean and has enough signal to learn from |
| **MODEL BENCHMARK** | [L24–34](../src/Evaluation/outputs/reporting.py#L24-L34) | All 3 models ranked by MAE, RMSE, R², with `[selected]` marker on the winner | Shows which model was chosen and how close the competition was |
| **BASELINE COMPARISON** | [L36–39](../src/Evaluation/outputs/reporting.py#L36-L39) | MAE/RMSE/R² if you just always guessed the mean wait time | Proves the ML model is better than "just guess the average" |
| **ROBUST EVALUATION** | [L41–52](../src/Evaluation/outputs/reporting.py#L41-L52) | Winner's full results: random split, chrono split, P90/P95/max errors | Shows model performance across different scenarios |
| **SEGMENT ERROR CHECKS** | [L54–58](../src/Evaluation/outputs/reporting.py#L54-L58) | Error on peak days/hours vs. non-peak | Exposes if the model is biased against certain scenarios |
| **WHY THESE MODELS** | [L60–63](../src/Evaluation/outputs/reporting.py#L60-L63) | Plain-English justification for Linear Regression, Random Forest, and Gradient Boosting | Documents the design decision for future reference |
| **WHY NOT OTHERS** | [L65–68](../src/Evaluation/outputs/reporting.py#L65-L68) | Why neural networks, scaling, one-hot encoding weren't used | Preempts "why didn't you try X?" questions |
| **PREPROCESSING CHOICES** | [L70–73](../src/Evaluation/outputs/reporting.py#L70-L73) | Why lag features, week_of_month, and filtering were chosen | Documents feature engineering rationale |
| **FEATURE IMPORTANCE** | [L75–77](../src/Evaluation/outputs/reporting.py#L75-L77) | Ranked list of all 16 features by importance score | Shows what the model actually relies on |

**How the benchmark section works** ([Lines 24–34](../src/Evaluation/outputs/reporting.py#L24-L34)):

```python
for _, row in model_rows.iterrows():
    marker = " [selected]" if bool(row["selected"]) else ""
    f.write(f"{row['model']}{marker}\n")
    f.write(f"  Train MAE: {row['train_mae']:.2f}\n")
    f.write(f"  Test MAE: {row['test_mae']:.2f}\n")
    f.write(f"  Chronological MAE: {row['chrono_test_mae']:.2f}\n")
    f.write(f"  CV MAE: {row['cv_mae']:.2f}\n")
    f.write(f"  Robust MAE: {row['robust_mae']:.2f}\n")
    f.write(f"  Test R2: {row['test_r2']:.4f}\n")
```

**Why the "WHY NOT" section matters:** In academic and production settings, reviewers ask "why didn't you use a neural network?" The report preemptively answers:
- No neural nets → dataset is too small, tabular data favors tree-based models
- No feature scaling → tree models are scale-insensitive by design
- No one-hot encoding → features are already numeric

> **Analogy:** This is the lab notebook of an experiment. Even if you get a great result, if you can't explain *why* you made each decision, the result isn't trustworthy.

---

### Step 9 — Sanity check predictions

**File:** [samples.py](../src/Evaluation/outputs/samples.py)

After saving the model, **6 hardcoded test cases** are run through it ([Lines 7–14](../src/Evaluation/outputs/samples.py#L7-L14)). These are scenarios where you already know the expected answer from the `TRUE_PATTERNS` table:

```python
test_cases = [
    {"name": "Monday 9am (Should be 55min)",    "hour": 9,  "day_of_week": 0, "is_peak_day": 1, "queue": 25},
    {"name": "Monday 10am (Should be 70min)",   "hour": 10, "day_of_week": 0, "is_peak_day": 1, "queue": 35},
    {"name": "Monday 8am (Should be 25min)",    "hour": 8,  "day_of_week": 0, "is_peak_day": 1, "queue": 8},
    {"name": "Wednesday 9am (Should be 19min)", "hour": 9,  "day_of_week": 2, "is_peak_day": 0, "queue": 10},
    {"name": "Wednesday 10am (Should be 25min)","hour": 10, "day_of_week": 2, "is_peak_day": 0, "queue": 12},
    {"name": "Wednesday 8am (Should be 9min)",  "hour": 8,  "day_of_week": 2, "is_peak_day": 0, "queue": 4},
]
```

For each test case, a feature row is manually constructed ([Lines 19–37](../src/Evaluation/outputs/samples.py#L19-L37)):

```python
feature_values = {
    "hour": case["hour"],
    "day_of_week": case["day_of_week"],
    "week_of_month": 2,                          # Fixed: mid-month (neutral)
    "month": sample_month,                        # Fixed: January
    "month_sin": float(np.sin(month_angle)),
    "month_cos": float(np.cos(month_angle)),
    "is_end_of_month": 0,                         # Not end of month
    "is_holiday": 0,                              # Not a holiday
    "queue_length_at_arrival": case["queue"],
    "queue_length_lag1": max(2, case["queue"] - 3),  # Estimated from queue
    "waiting_time_lag1": case["lag_wait"],
    # ... all 16 features filled in ...
}
row = [feature_values[name] for name in features]  # Same order as training
test_X = pd.DataFrame([row], columns=features)
pred = model.predict(test_X)[0]
```

**Why these specific test cases?**

| Test | Expected | What it tests |
|---|---|---|
| Monday 9am, 25 in queue | ~55 min | Peak day + peak hour + high queue → model should predict HIGH |
| Monday 10am, 35 in queue | ~70 min | The busiest moment in the entire week |
| Monday 8am, 8 in queue | ~25 min | Peak day but first hour → should be moderate, not extreme |
| Wednesday 9am, 10 in queue | ~19 min | Mid-week + peak hour + low queue → model should predict LOW |
| Wednesday 10am, 12 in queue | ~25 min | Slightly busier mid-week hour |
| Wednesday 8am, 4 in queue | ~9 min | Calmest combination in the dataset |

**What to look for:** If the model predicts Monday 9am at 55 min but Wednesday 8am also at 55 min, the model failed to learn the day/hour patterns. If Monday 9am predicts 9 min, the model is essentially broken.

> **Analogy:** After a student finishes an exam prep program, you give them 6 practice questions where you already know the answer. If they get 5/6 right, the program worked. If they get 1/6 right, something went wrong in the teaching.

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
4. [src/Evaluation/model_quality/model_evaluation.py](../src/Evaluation/model_quality/model_evaluation.py) — See the 3-way evaluation strategy and how `robust_mae` is computed.
5. [src/model_implementation/train_model.py](../src/model_implementation/train_model.py) — The full training pipeline in one place.
6. [src/Prediction/inference.py](../src/Prediction/inference.py) — How a prediction is made at query time.
7. [src/Prediction/cli.py](../src/Prediction/cli.py) — How the interface presents results to users.

---

## 🌐 Backend API — Weekly Forecast Endpoint

### `GET /api/weekly-forecast`

**File:** [Backend/app.py](../Backend/app.py) — `api_weekly_forecast()`

Returns a Monday–Saturday ML forecast for the week containing the provided date. Each day entry includes overall average wait, congestion level, best hour to visit, and worst hour to avoid — computed by running the Monte Carlo simulation across all 9 working hours.

**Query parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `date` | `YYYY-MM-DD` string | today | Any date within the target week |

**How it works:**

```python
# 1. Snap to Monday of the week (regardless of which day date falls on)
monday = target - pd.Timedelta(days=target.dayofweek)

# 2. For each day Mon–Sat, for each hour 8am–4pm:
mc = _monte_carlo_predict(day_date, day_name, hour, mc_runs=500)

# 3. Aggregate hourly results into daily summary
overall   = mean of all hourly mean predictions
best_time = hour with lowest mean prediction
worst_time = hour with highest mean prediction
congestion = LOW / MODERATE / HIGH based on overall average
```

**Holiday handling:** If a day falls on a Philippine holiday (checked against `_holiday_md`), that day returns `congestion: "CLOSED"` and no prediction data.

**Example response:**

```json
{
  "weekLabel": "May 18 – May 23, 2026",
  "weekOf": "2026-05-18",
  "days": [
    {
      "date": "2026-05-18",
      "dayName": "Monday",
      "shortDate": "May 18",
      "isHoliday": false,
      "overall": 50.4,
      "congestion": "HIGH",
      "bestTime": "08:00",
      "bestWait": 27.1,
      "bestP10": 25.0,
      "bestP90": 29.0,
      "worstTime": "14:00",
      "worstWait": 66.2,
      "hourly": [
        { "hour": "08:00", "wait": 27.1, "p10": 25.0, "p90": 29.0 },
        ...
      ]
    },
    ...
  ]
}
```

---

## 🖥️ Frontend — WeeklyForecastSection Component

**File:** [Frontend/src/components/landing/WeeklyForecastSection.tsx](../Frontend/src/components/landing/WeeklyForecastSection.tsx)

A new landing page section placed directly below the Live Simulation & Demo section. Lets users select any week via a calendar and instantly see a Monday–Saturday forecast from the ML model.

### Layout

```
┌─────────────────────────────────────────────────────────┐
│                   Weekly Forecast                        │
│          (header + subtitle)                            │
├──────────────────────┬──────────────────────────────────┤
│  Calendar (month     │  Week at a Glance panel          │
│  view, click any     │  (quick summary list Mon–Sat)    │
│  day to select week) │                                  │
│  ← Selected week →   │                                  │
└──────────────────────┴──────────────────────────────────┘
│  Mon  │  Tue  │  Wed  │  Thu  │  Fri  │  Sat  │  ← Day cards
│ card  │ card  │ card  │ card  │ card  │ card  │
└───────┴───────┴───────┴───────┴───────┴───────┘
```

### Key implementation details

**Timezone-safe date formatting:**

```typescript
function toYMD(d: Date): string {
  // Local date components — NOT toISOString() which converts to UTC
  // toISOString() would shift May 11 midnight (+08:00) → May 10 UTC
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}
```

**Week snapping (mirrors backend logic):**

```typescript
function getMondayOfWeek(date: Date): Date {
  const d = new Date(date);
  const day = d.getDay(); // 0=Sun, 1=Mon …
  const diff = day === 0 ? -6 : 1 - day;
  d.setDate(d.getDate() + diff);
  return d;
}
```

**Calendar grid:** Built from the first day of the displayed month, offset by `firstDay.getDay()` to align with the Sunday-start column layout. Cells outside the month are rendered as invisible placeholders.

**Selected week highlight:** `isSameWeek()` checks if each calendar cell belongs to the currently selected week by comparing `getMondayOfWeek(cell)` with `selectedMonday`.

**API call:** Uses `fetchWeeklyForecast(toYMD(selectedMonday))` from `Frontend/src/lib/api.ts`, which hits `GET /api/weekly-forecast?date=<date>`.

### Congestion color mapping

| Congestion | Card gradient | Badge color | Dot |
|---|---|---|---|
| HIGH | `from-red-600/30` | `bg-red-600` | red |
| MODERATE | `from-yellow-600/30` | `bg-yellow-600` | yellow |
| LOW | `from-green-600/30` | `bg-green-600` | green |
| CLOSED | `from-gray-700/30` | `bg-gray-600` | gray |

---

## 🔄 Dynamic Date Detection — Live Simulation

**File:** [Frontend/src/lib/api.ts](../Frontend/src/lib/api.ts) — `getDateForDayThisWeek()`

The Live Simulation section previously used hardcoded dates (`Monday: '2026-05-11'` etc.). These were replaced with a function that computes the **real calendar date** for the selected day within the current week:

```typescript
function getDateForDayThisWeek(dayName: string): string {
  const ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
  const today = new Date();
  const dow = today.getDay();              // 0=Sun, 1=Mon …
  const toMonday = dow === 0 ? -6 : 1 - dow;
  const monday = new Date(today);
  monday.setDate(today.getDate() + toMonday);
  monday.setHours(0, 0, 0, 0);

  const idx = ORDER.indexOf(dayName);
  const target = new Date(monday);
  target.setDate(monday.getDate() + (idx >= 0 ? idx : 0));

  // Local date components — avoids UTC shift bug
  return `${y}-${m}-${d}`;
}
```

**Why this matters:** If a user runs the Live Simulation on a Monday, selecting "Monday" now sends that actual Monday's date to the backend. This enables correct holiday detection and accurate `week_of_month` feature values instead of always using a fixed May 2026 date.

