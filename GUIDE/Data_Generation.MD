# Data_.py — Study Script
### iQueue Synthetic Data Generator | Full Walkthrough

---

## WHAT IS THIS FILE?

`Data_.py` is a **synthetic data factory** for the iQueue system —
an AI that predicts wait times at the LTO CDO branch.

Because real queue logs don't exist, this script **invents** realistic
transaction data. But not randomly — it uses 17+ mathematical modifiers
that mirror how real queues actually behave.

**It does two things:**
1. Generates ~28,000 fake but realistic transactions
2. Does light feature engineering (lag features, binary flags, cyclical encoding)

**It does NOT predict anything.** Prediction is the model's job in `main.py`.

---

## HOW IT CONNECTS TO THE PIPELINE

```
Data_.py  →  CSV on disk  →  main.py preprocessing  →  model training  →  predictions
```

- `Data_.py` is the ONLY script that writes to disk
- Everything downstream reads that one CSV and processes in-memory
- Once the model is trained, `Data_.py` is retired — its job is done

---

## THE RANDOM SEED

```python
np.random.seed(42)
```

This locks the random number generator. Run the script 100 times —
you get identical data every time.

**Why it matters:** If training data changed every run, you couldn't
compare model versions fairly. The seed makes results reproducible.

---

## TRUE_PATTERNS — The Ground Truth

```python
TRUE_PATTERNS = {
    'Monday':    [25, 55, 70, 60, 25, 50, 65, 55, 35],
    'Wednesday': [ 9, 19, 25, 18,  9, 15, 22, 18, 12],
    'Friday':    [22, 50, 65, 55, 26, 48, 60, 51, 32],
    ...
}
```

Each list = 9 values = one per working hour: [8am, 9am, 10am, 11am, 12pm, 1pm, 2pm, 3pm, 4pm]

These are **base wait times in minutes** before any modifier is applied.

**Key observations:**
- Monday 10am = 70 min → single busiest slot of the week
- Wednesday 12pm = 9 min → lunch dip on the calmest day
- All days dip at 12pm → fewer people come during lunch
- These are the "signal" the model is trained to learn

---

## THE HOLIDAY PARSER — load_ph_holidays()

Reads a Philippine holiday calendar CSV and returns a **set** of dates.

```python
holiday_dates = load_ph_holidays(HOLIDAY_CALENDAR_PATH, START_DATE.year)
```

**Why a set?** Checking `date in holiday_dates` happens ~28,000 times.
A set does this in O(1) — instant. A list would scan every element each time.

**Defensive design:** If the calendar file is missing, it returns an
empty set and the script continues without holiday effects. No crash.

**How it parses:** Uses regex to find patterns like "Jan 1 :" in the file,
extracts month name and day number, converts to datetime.date objects.

---

## THE 17 MODIFIERS

All modifiers are **multipliers** stacked on top of the base pattern value.

```
final_wait = base_pattern × mod1 × mod2 × mod3 × ... × mod17
```

### GROUP 1 — Weekly / Long-term

**1. Long-term trend**
```python
1.0 + 0.01*sin(2π*week/NUM_WEEKS) + 0.006*(week/NUM_WEEKS)
```
Sine wave + slow linear growth. Makes volume slowly evolve over 10 months.
Without this, every week would be a perfect copy of the last.

---

### GROUP 2 — Daily Modifiers

**2. Seasonal variation**
```python
1.0 + 0.08*sin(2π*(month-1)/12)
```
Summer months push wait times ~8% higher. Creates a natural annual cycle.

**3. End-of-month rush**
```python
1.06 if day >= last_day - 2 else 1.0
```
Last 3 days of each month = +6% wait time.
People rush to renew licenses before month-end deadlines.

**4. Holiday reduction**
```python
0.75 if is_holiday else 1.0
```
ON a holiday, fewer people come in → wait times drop 25%.

**5. Pre-holiday surge**
```python
1.12 if tomorrow in holiday_dates else 1.0
```
The day BEFORE a holiday → +12%.
People rush knowing the office will be closed tomorrow.

**6. Multi-frequency day wave**
```python
1.0 + 0.03*sin(2π*day_idx/9) + 0.02*cos(2π*day_idx/17)
```
Two overlapping sine waves with periods of 9 and 17 days.
They never perfectly repeat → realistic day-to-day texture.
Without this, every Monday would look identical.

**7. Autocorrelated day load**
```python
day_load = 0.6 * prev_day_load + 0.4 * baseline_load
```
60% of today's load comes from yesterday.
Busy Mondays tend to make Tuesdays busier too.
`prev_day_load` is a state variable that survives between days.

**8. Baseline load wave**
```python
1.0 + 0.04*sin(2π*day_idx/11) + 0.03*cos(2π*day_idx/23)
```
The "natural" daily load before autocorrelation.
Periods of 11 and 23 days — prime-like so they don't align, creating
complex non-repeating patterns that feed into modifier #7.

**9. Day-of-week capacity**
```python
1.03 if Monday or Friday
0.98 if Tuesday, Wednesday, Thursday
1.00 if Saturday
```
Amplifies Mon/Fri peaks. Dampens mid-week.
Sharpens the signal so the model learns day-of-week more strongly.

---

### GROUP 3 — Hourly Modifiers

**10. Hourly micro-wave**
```python
1.0 + 0.025*sin(2π*(hour_idx + day_idx)/9)
```
±2.5% variation within each hour slot.
Prevents every 8am Monday from having exactly the same wait time.

---

### GROUP 4 — Per-Transaction Modifiers

**11. Multiplicative noise**
```python
wait_variation = uniform(0.85, 1.15)
```
Each transaction gets a random ±15% multiplier.
No two transactions are identical even in the same hour.

**12. Additive noise**
```python
noise = uniform(-0.08, 0.08)
wait_time = wait_time * (1 + noise)
```
An additional ±8% layer on top of modifier #11.
Two separate noise sources create a more realistic spread.

**13. Wait time clamp**
```python
wait_time = max(5, min(90, wait_time))
```
Hard floor of 5 min. Hard ceiling of 90 min.
Prevents compounding multipliers from producing absurd values.

**14. Queue length ↔ wait time correlation**
```python
if wait > 45:  queue = randint(25, 45)   # HIGH congestion
elif wait > 25: queue = randint(12, 28)  # MODERATE
else:           queue = randint(2, 12)   # LOW
```
Queue length is NOT independent of wait time — they're correlated.
The model learns: high queue = high wait. This mirrors reality.

**15. Nonlinear congestion amplification**
```python
if queue > 18:
    wait_time *= 1.0 + (queue - 18) / 90
```
Once the queue exceeds 18 people, each extra person makes wait grow
faster than linearly. Simulates real bottleneck effects.

**16. Holiday transaction count reduction**
```python
randint(4, 9)   if holiday    # fewer people
randint(8, 15)  if normal     # normal volume
```
On holidays: not just shorter waits — fewer transactions per hour too.
Both volume AND wait time are affected.

**17. Service time ↔ congestion correlation**
```python
if wait > 45:  service = uniform(45, 75)  # complex transactions
elif wait > 25: service = uniform(30, 50)  # medium
else:           service = uniform(15, 35)  # quick
```
Busier periods have longer service times.
Peak hours attract more complex transactions (new licenses, renewals
with issues). Off-peak hours are mostly quick queries.

---

## HOW waiting_time_min IS GENERATED

It's built from scratch by stacking all modifiers:

```
base (TRUE_PATTERNS[day][hour])
  × seasonal_factor
  × eom_factor
  × holiday_factor
  × pre_holiday_factor
  × trend_factor
  × day_wave
  × hour_wave
  × day_load
  × capacity_factor
  → × per-transaction noise (×1.15 or ×0.85 etc.)
  → clamped to [5, 90]
  = waiting_time_min  ← stored in CSV
```

**Example: Monday 10am**
```
base         = 70
× seasonal   = 70 × 1.05  = 73.5
× eom        = 73.5 × 1.06 = 77.9
× trend      = 77.9 × 1.008 = 78.5
× waves+load = 78.5 × 1.05 = 82.4
× noise      = 82.4 × 0.97 = 79.9  ← final waiting_time_min
```

---

## HOW service_time AND total_time RELATE TO waiting_time_min

`service_time_min` is derived FROM `waiting_time_min`:
```python
if wait > 45:   service = uniform(45, 75)
elif wait > 25: service = uniform(30, 50)
else:           service = uniform(15, 35)
```

`total_time_in_system_min` is simply:
```python
total_time = wait_time + service_time
```

**Why they exist in the CSV if the model predicts waiting_time_min:**

They exist for FLEXIBILITY. In main.py, the model drops whichever
columns it doesn't need based on what it's predicting:

| Predicting...              | Drop these columns                          |
|----------------------------|---------------------------------------------|
| waiting_time_min           | service_time_min, total_time_in_system_min  |
| service_time_min           | waiting_time_min, total_time_in_system_min  |
| total_time_in_system_min   | waiting_time_min, service_time_min          |

Data_.py is the kitchen — it preps all ingredients.
main.py is the chef — it decides which ones go into the dish.

---

## FEATURE ENGINEERING

After generation, the script engineers new columns from existing ones.

### 1. month_sin and month_cos
**Source:** `month` (raw integer 1–12)

```python
month_angle = 2 * np.pi * (month - 1) / 12
month_sin   = np.sin(month_angle)
month_cos   = np.cos(month_angle)
```

**The problem with raw month:** The model sees numbers.
December = 12, January = 1 → looks like they are 11 apart.
But December and January are calendar neighbors.

**The fix — put months on a circle (like a clock):**

```
        Jan (12 o'clock)
   Dec       Feb
Nov               Mar
Oct               Apr
   Sep       May
        Aug (6 o'clock)
```

| month      | month_sin | month_cos | distance from Jan |
|------------|-----------|-----------|-------------------|
| Jan (1)    |  0.000    |  1.000    | 0.00 (itself)     |
| Dec (12)   | -0.259    |  0.966    | 0.26 ← very close |
| Jul (7)    |  0.000    | -1.000    | 2.00 ← farthest   |

December and January are only 0.26 units apart mathematically.
Without this encoding, the model thinks they're 11 apart.

**Why two values (sin AND cos)?**
One value alone can't uniquely identify every month.
April and August both have month_sin ≈ 0.866 — you need cos too
to tell them apart. Together they always give a unique pair.

---

### 2. is_end_of_month
**Source:** `date`

```python
last_day = calendar.monthrange(year, month)[1]
is_end_of_month = 1 if day_date.day >= last_day - 2 else 0
```

**This is a binary feature — only two possible values:**

- **0 = NO** → normal day, no deadline rush, queue behaves normally
- **1 = YES** → one of the last 3 days of the month, +6% wait boost

**Example — January (31 days):**

| date   | day | last_day - 2 | is_end_of_month |
|--------|-----|-------------|-----------------|
| Jan 27 | 27  | 29          | 0 — normal      |
| Jan 29 | 29  | 29          | 1 — rush starts |
| Jan 31 | 31  | 29          | 1 — last day    |

The model doesn't need to figure out that Jan 29–31 and
Feb 26–28 and Mar 29–31 are ALL end-of-month rushes.
This flag makes it one clear signal instead of 12 date patterns.

---

### 3. is_peak_day
**Source:** `day_name`

```python
is_peak_day = 1 if day_name in ['Monday', 'Friday'] else 0
```

| day_name  | base wait 10am | is_peak_day |
|-----------|---------------|-------------|
| Monday    | 70 min        | 1           |
| Tuesday   | 32 min        | 0           |
| Wednesday | 25 min        | 0           |
| Thursday  | 32 min        | 0           |
| Friday    | 65 min        | 0           |
| Saturday  | 33 min        | 0           |

Without this, the model has to discover on its own that
day=0 and day=4 are both special. This flag hands it that
insight directly so it learns faster.

---

### 4. is_peak_hour
**Source:** `hour`

```python
is_peak_hour = 1 if hour in [9, 10, 11, 14, 15] else 0
```

Two clusters of peak hours:
- Morning rush: 9am, 10am, 11am
- Afternoon rush: 2pm, 3pm

| hour | Mon base wait | is_peak_hour |
|------|--------------|--------------|
| 8am  | 25 min       | 0            |
| 9am  | 55 min       | 1            |
| 10am | 70 min       | 1            |
| 11am | 60 min       | 1            |
| 12pm | 25 min       | 0 ← lunch dip|
| 2pm  | 65 min       | 1            |
| 3pm  | 55 min       | 1            |
| 4pm  | 35 min       | 0            |

**Most powerful when combined:**
is_peak_day=1 AND is_peak_hour=1 → Monday 10am → worst case ~70+ min

---

### 5. is_weekend
**Source:** `day_of_week`

```python
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
```

Saturday (day_of_week=5) = 1. All weekdays = 0.
Sunday is never generated — LTO is closed.

Saturday has fundamentally different patterns (half-day,
different customer types). This flag makes that difference explicit.

---

### 6. queue_length_lag1 and waiting_time_lag1
**Source:** previous row's `queue_length_at_arrival` and `waiting_time_min`

```python
df['queue_length_lag1'] = df.groupby('date')['queue_length_at_arrival'].shift(1)
df['waiting_time_lag1'] = df.groupby('date')['waiting_time_min'].shift(1)
```

`.shift(1)` moves the column DOWN by 1 row — each transaction
gets the PREVIOUS transaction's values.

`groupby('date')` resets the shift at every new day —
yesterday's last transaction never bleeds into today's first.

**Before lag:**
| txn | arrival | queue | wait  |
|-----|---------|-------|-------|
| #1  | 8:12am  | 4     | 12.0  |
| #2  | 8:34am  | 7     | 18.3  |
| #3  | 9:05am  | 15    | 42.1  |
| #4  | 9:22am  | 20    | 58.4  |
| #5  | 9:47am  | 24    | 65.2  |

**After lag:**
| txn | arrival | queue | wait  | queue_lag1      | wait_lag1       |
|-----|---------|-------|-------|-----------------|-----------------|
| #1  | 8:12am  | 4     | 12.0  | 8 ← default     | 25 ← default    |
| #2  | 8:34am  | 7     | 18.3  | 4 ← from txn #1 | 12.0 ← txn #1   |
| #3  | 9:05am  | 15    | 42.1  | 7 ← from txn #2 | 18.3 ← txn #2   |
| #4  | 9:22am  | 20    | 58.4  | 15 ← from txn #3| 42.1 ← txn #3   |
| #5  | 9:47am  | 24    | 65.2  | 20 ← from txn #4| 58.4 ← txn #4   |

**Transaction #1 has no previous transaction.**
It gets day-type defaults:
- Monday/Friday → queue_lag1=8, wait_lag1=25 (busy mornings)
- Other days    → queue_lag1=3, wait_lag1=10 (calm mornings)

**In plain English:**
- `queue_length_lag1 = 15` means: "The person just before me found 15 people already in the queue."
- `waiting_time_lag1 = 42.1` means: "The person just before me waited 42 minutes before being served."

**Why this is the most powerful feature:**
Same day. Same hour. Totally different predictions:

> Monday 9am + wait_lag1=12 → queue is still calm → predict ~35 min
> Monday 9am + wait_lag1=58 → queue is exploding → predict ~70 min

Without lag features, both cases look identical to the model
and it outputs the same prediction for both. The lag gives
the model momentum awareness — it knows what's happening
RIGHT NOW, not just what usually happens at this time.

---

### 7. total_time_in_system_min
**Source:** `waiting_time_min` + `service_time_min`

```python
total_time = round(wait_time + service_time, 1)
```

**Example:**
```
wait_time    = 79.9 min  (time in queue)
service_time = 61.3 min  (time at the counter)
total_time   = 141.2 min (full experience)
```

Answers the real customer question:
"How long will I actually be inside this building total?"

---

## WHY NOT JUST USE RANDOM DATA?

Random data would teach the model nothing.

If wait times were uniformly random between 5–90 min with no pattern,
the model would learn that all times and days are equally likely.
It would then predict ~47 min (the mean) for every single input — useless.

The 17 modifiers embed **learnable structure** so the model can answer:
- "Monday 10am" → ~65–75 min
- "Wednesday 12pm" → ~9–12 min
- "Friday 3pm, wait_lag1=58" → ~70+ min

---

## SUMMARY — What each feature does for the model

| Feature               | Source          | What it tells the model                        |
|-----------------------|-----------------|------------------------------------------------|
| month_sin, month_cos  | month           | Are these months seasonally similar?           |
| is_end_of_month       | date            | Is there a deadline rush today?                |
| is_peak_day           | day_name        | Is this one of the two busiest days?           |
| is_peak_hour          | hour            | Am I inside a rush window right now?           |
| is_weekend            | day_of_week     | Is this a fundamentally different day type?    |
| queue_length_lag1     | prev row queue  | How congested was it just before I arrived?    |
| waiting_time_lag1     | prev row wait   | Is the queue currently getting worse?          |
| total_time_in_system  | wait + service  | How long is the full customer experience?      |

**The pattern across all of them:**
Take something the model would struggle to figure out on its own,
and hand it the answer directly as a new column.

---

*End of study script — Data_.py | iQueue synthetic data generator*