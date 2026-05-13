# iQueue Code Reference (Fully Detailed Bullets)

This guide is intentionally verbose. Every bullet below explains what the code does, why it matters, and how it connects to other files.

## File Index

- [main.py](../main.py): Orchestrator entrypoint that runs the training pipeline first and then starts the prediction CLI; this guarantees the model artifact exists before inference begins.
- [data/Data_.py](../data/Data_.py): Synthetic data generator that creates realistic queue transactions with temporal and holiday effects; this file is the source of the dataset consumed by training and prediction pattern baselines.
- [src/Preprocessing/preprocess.py](../src/Preprocessing/preprocess.py): Thin compatibility wrapper that re-exports the refactored preprocessing helpers.
- [src/Preprocessing/loader.py](../src/Preprocessing/loader.py): Loads raw CSVs and applies feature engineering/cleaning.
- [src/Preprocessing/features.py](../src/Preprocessing/features.py): Defines canonical feature order and builds feature DataFrames for training/inference.
- [src/Preprocessing/calendar.py](../src/Preprocessing/calendar.py): Holiday parsing helper shared by preprocessing and prediction.
- [src/Prediction/predict.py](../src/Prediction/predict.py): User-facing forecasting CLI entrypoint (re-exports refactored prediction modules).
- [src/Prediction/context.py](../src/Prediction/context.py): Loads model/data and builds date-aware historical pattern maps.
- [src/Prediction/patterns.py](../src/Prediction/patterns.py): Pattern-map builders and lookup fallbacks.
- [src/Prediction/inference.py](../src/Prediction/inference.py): Feature assembly and wait-time prediction routines.
- [src/Prediction/cli.py](../src/Prediction/cli.py): Interactive console UI for weekly/daily/best-time views.
- [src/model_implementation/__init__.py](../src/model_implementation/__init__.py): Package-level export file that re-exports model catalog utilities to simplify imports in the training script.
- [src/model_implementation/train_model.py](../src/model_implementation/train_model.py): Training pipeline orchestrator that calls the modular evaluation/report/plot helpers.
- [src/model_implementation/evaluation.py](../src/model_implementation/evaluation.py): Model evaluation logic and data-quality checks.
- [src/model_implementation/metrics.py](../src/model_implementation/metrics.py): Metric helpers shared across training and baseline evaluation.
- [src/model_implementation/plots.py](../src/model_implementation/plots.py): Plot generators for distribution, heatmaps, and comparisons.
- [src/model_implementation/reporting.py](../src/model_implementation/reporting.py): Metrics report writer for training runs.
- [src/model_implementation/splits.py](../src/model_implementation/splits.py): Chronological split helper for time-aware evaluation.
- [src/model_implementation/samples.py](../src/model_implementation/samples.py): Sample-case prediction sanity checks.
- [src/model_implementation/model_zoo/__init__.py](../src/model_implementation/model_zoo/__init__.py): Catalog builder that defines which model builders are active candidates during benchmarking.
- [src/model_implementation/model_zoo/linear_regression.py](../src/model_implementation/model_zoo/linear_regression.py): Baseline model builder used to establish a simple linear reference point for comparison.
- [src/model_implementation/model_zoo/random_forest.py](../src/model_implementation/model_zoo/random_forest.py): RandomForest builder with robust tabular defaults and optional parameter overrides for experimentation.
- [src/model_implementation/model_zoo/gradient_boosting.py](../src/model_implementation/model_zoo/gradient_boosting.py): GradientBoosting builder used as a second nonlinear ensemble candidate with different inductive behavior than RandomForest.
- [src/model_implementation/model_zoo/extra_trees.py](../src/model_implementation/model_zoo/extra_trees.py): ExtraTrees builder prepared for future catalog inclusion; currently defined but not active in the default model catalog.

## File Intros (What Each File Does)

### [main.py](../main.py)

- Purpose: Acts as a one-command launcher so users can run the full pipeline without remembering separate training and prediction commands.
- Primary behavior: Executes [src/model_implementation/train_model.py](../src/model_implementation/train_model.py) first, then executes [src/Prediction/predict.py](../src/Prediction/predict.py), preserving workflow order.
- Why this matters: Prevents common runtime failures where prediction is attempted before a model exists.
- Upstream dependencies: Relies on Python runtime and successful imports in the downstream scripts.
- Downstream effect: Produces a trained model file through training, then immediately opens the interactive forecast interface.

### [data/Data_.py](../data/Data_.py)

- Purpose: Generates synthetic yet structured queue data when real operational logs are unavailable or incomplete.
- Primary behavior: Simulates arrival transactions over weeks, days, and hours while applying seasonal, holiday, and congestion effects.
- Why this matters: Gives the model realistic patterns to learn (weekday differences, peak-hour behavior, lag dynamics).
- Key output: Writes [data/synthetic_lto_cdo_queue_90days.csv](../data/synthetic_lto_cdo_queue_90days.csv), which is the backbone dataset used by training and prediction baselines.
- Downstream effect: Supplies raw rows for [src/Preprocessing/loader.py](../src/Preprocessing/loader.py) and historical pattern statistics for [src/Prediction/context.py](../src/Prediction/context.py).

### [src/Preprocessing/preprocess.py](../src/Preprocessing/preprocess.py)

- Purpose: Centralizes feature construction so training and inference semantics stay consistent.
- Primary behavior: Delegates to loader/features/calendar modules to parse dates, add holiday flags, add cyclic month encodings, derive week-of-month, and clean invalid rows.
- Why this matters: Reduces drift between model training assumptions and what inference expects.
- Key contract: Feature order returned here must match inference feature order in [src/Prediction/inference.py](../src/Prediction/inference.py).
- Downstream effect: Feeds clean feature matrices into [src/model_implementation/train_model.py](../src/model_implementation/train_model.py).

### [src/model_implementation/train_model.py](../src/model_implementation/train_model.py)

- Purpose: Owns all training, evaluation, model selection, and artifact writing logic.
- Primary behavior: Loads engineered data, evaluates multiple models across random split + chronological split + cross-validation, then persists best model.
- Why this matters: Gives a robust quality signal instead of over-trusting one split.
- Key outputs: [models/queue_model.pkl](../models/queue_model.pkl), [outputs/model_comparison.csv](../outputs/model_comparison.csv), [outputs/metrics.txt](../outputs/metrics.txt), and plot files in [outputs/plots](../outputs/plots).
- Downstream effect: Saved model and reports are consumed by [src/Prediction/predict.py](../src/Prediction/predict.py) and by human review workflows.

### [src/Prediction/predict.py](../src/Prediction/predict.py)

- Purpose: Converts model outputs and historical patterns into actionable user guidance.
- Primary behavior: Loads model and data, builds fallback pattern maps, runs prediction routines, and renders interactive weekly/daily/best-time views.
- Why this matters: Users need interpretable recommendations, not raw regression outputs.
- Inputs: Depends on trained artifact [models/queue_model.pkl](../models/queue_model.pkl) and historical dataset [data/synthetic_lto_cdo_queue_90days.csv](../data/synthetic_lto_cdo_queue_90days.csv).
- Downstream effect: Produces console-level decision support and congestion insights.

### [src/model_implementation/__init__.py](../src/model_implementation/__init__.py)

- Purpose: Provides a clean package API surface.
- Primary behavior: Re-exports catalog builder so training code can import from package root.
- Why this matters: Keeps import paths simpler and reduces coupling to folder depth.
- Downstream effect: Used by [src/model_implementation/train_model.py](../src/model_implementation/train_model.py) for model catalog access.

### [src/model_implementation/model_zoo/__init__.py](../src/model_implementation/model_zoo/__init__.py)

- Purpose: Defines active benchmark candidates in one place.
- Primary behavior: Returns a dictionary of model name -> instantiated estimator builders.
- Why this matters: Makes experimentation easy by changing one file when adding/removing models.
- Downstream effect: Directly consumed by [src/model_implementation/train_model.py](../src/model_implementation/train_model.py) in model-evaluation loop.

### [src/model_implementation/model_zoo/linear_regression.py](../src/model_implementation/model_zoo/linear_regression.py)

- Purpose: Supplies a minimal linear baseline candidate.
- Primary behavior: Returns `LinearRegression()` with default settings.
- Why this matters: Baseline performance helps quantify value from nonlinear ensembles.
- Downstream effect: Included in active catalog and benchmarked during training.

### [src/model_implementation/model_zoo/random_forest.py](../src/model_implementation/model_zoo/random_forest.py)

- Purpose: Supplies a strong nonlinear ensemble candidate.
- Primary behavior: Builds `RandomForestRegressor` with tuned defaults and optional override dictionary.
- Why this matters: Handles nonlinear interactions and noisy tabular relationships well.
- Downstream effect: Often a top-performing candidate in benchmark selection.

### [src/model_implementation/model_zoo/gradient_boosting.py](../src/model_implementation/model_zoo/gradient_boosting.py)

- Purpose: Supplies an alternative ensemble strategy for comparison.
- Primary behavior: Builds `GradientBoostingRegressor` with moderate learning rate and subsampling settings.
- Why this matters: Captures residual structure differently than bagging models.
- Downstream effect: Provides diversity in candidate behavior for robust model selection.

### [src/model_implementation/model_zoo/extra_trees.py](../src/model_implementation/model_zoo/extra_trees.py)

- Purpose: Provides a ready-to-enable experimental candidate.
- Primary behavior: Builds `ExtraTreesRegressor` with configured defaults.
- Why this matters: Keeps extension path low-friction for future experiments.
- Downstream effect: Can be added to [src/model_implementation/model_zoo/__init__.py](../src/model_implementation/model_zoo/__init__.py) when needed.

## Restart From Here (Slower Walkthrough)

### Step 1: Pipeline entry

- [main.py#L1-L2](../main.py#L1-L2): Imports process/system modules so downstream scripts can be launched with the same interpreter context, reducing environment mismatch risk.
- [main.py#L5](../main.py#L5): Prints startup text to indicate orchestration has begun, which helps operators distinguish startup failures from training failures.
- [main.py#L7](../main.py#L7): Executes training before prediction; this ordering ensures model artifacts are current and available.
- [main.py#L9](../main.py#L9): Prints handoff message to mark phase transition from batch training to interactive forecasting.
- [main.py#L10](../main.py#L10): Starts prediction CLI only after successful training, closing the workflow loop for end users.

### Step 2: Data generator setup

- [data/Data_.py#L1-L7](../data/Data_.py#L1-L7): Imports utilities needed for simulation math, date handling, parsing, and tabular output.
- [data/Data_.py#L9](../data/Data_.py#L9): Seeds random number generation so results are reproducible and debugging can compare runs fairly.
- [data/Data_.py#L12-L16](../data/Data_.py#L12-L16): Defines period and file paths that control simulation scope and output destination.
- [data/Data_.py#L19-L26](../data/Data_.py#L19-L26): Declares core weekday/hour target patterns that represent intended operational behavior.

### Step 3: Holiday parsing

- [data/Data_.py#L31-L54](../data/Data_.py#L31-L54): Converts holiday text to date set for fast membership checks during simulation.
- [data/Data_.py#L41](../data/Data_.py#L41): Iterates through each source line so parser handles calendar file as semi-structured text.
- [data/Data_.py#L42](../data/Data_.py#L42): Uses regex extraction to map month/day fragments into normalized numeric values.
- [data/Data_.py#L50-L53](../data/Data_.py#L50-L53): Safely skips malformed dates, preventing generator crashes and preserving run continuity.
- [data/Data_.py#L56](../data/Data_.py#L56): Stores parsed holiday dates for repeated use by daily/hourly simulation logic.

### Step 4: Core generation loops

- [data/Data_.py#L62](../data/Data_.py#L62): Week loop builds medium-term variation and trend effects across the simulation horizon.
- [data/Data_.py#L67](../data/Data_.py#L67): Day loop restricts generation to working days and applies weekday-specific behavior.
- [data/Data_.py#L96](../data/Data_.py#L96): Hour loop introduces intraday pattern shape (morning peaks, midday dips, etc.).
- [data/Data_.py#L110](../data/Data_.py#L110): Transaction loop creates row-level observations for model learning granularity.

### Step 5: Wait-time construction

- [data/Data_.py#L77-L86](../data/Data_.py#L77-L86): Computes seasonal, month-end, and holiday multipliers that create realistic temporal drift.
- [data/Data_.py#L88-L93](../data/Data_.py#L88-L93): Applies day-load/capacity coupling so nearby dates share plausible operational momentum.
- [data/Data_.py#L97-L99](../data/Data_.py#L97-L99): Combines baseline pattern and multipliers into base_wait used by per-transaction simulation.

### Step 6: Row simulation details

- [data/Data_.py#L111-L113](../data/Data_.py#L111-L113): Creates concrete arrival timestamps, enabling sort order and lag feature derivation later.
- [data/Data_.py#L115-L119](../data/Data_.py#L115-L119): Adds bounded noise and clipping so values vary naturally without unrealistic extremes.
- [data/Data_.py#L122-L127](../data/Data_.py#L122-L127): Maps wait levels to queue-size ranges, preserving correlation between queue and wait.
- [data/Data_.py#L130-L132](../data/Data_.py#L130-L132): Adds nonlinear congestion amplification for long queues, improving realism.
- [data/Data_.py#L135-L140](../data/Data_.py#L135-L140): Sets service-time ranges conditional on congestion state.
- [data/Data_.py#L142-L160](../data/Data_.py#L142-L160): Appends finalized row dictionary with all fields needed downstream.

### Step 7: Post-processing

- [data/Data_.py#L163](../data/Data_.py#L163): Converts accumulated records into DataFrame for vectorized transforms.
- [data/Data_.py#L166](../data/Data_.py#L166): Sorts chronologically, which is required for meaningful lag calculation.
- [data/Data_.py#L169-L170](../data/Data_.py#L169-L170): Builds lag features per date group to encode short-term dynamics.
- [data/Data_.py#L173-L184](../data/Data_.py#L173-L184): Initializes first-row lag defaults to avoid null edge cases at day boundaries.
- [data/Data_.py#L186](../data/Data_.py#L186): Clears any residual missing values for training safety.
- [data/Data_.py#L189](../data/Data_.py#L189): Adds weekend indicator used by model features.
- [data/Data_.py#L192](../data/Data_.py#L192): Persists dataset for preprocess/training/pattern lookup consumers.

### Step 8: Verification output meaning

- [data/Data_.py#L58-L60](../data/Data_.py#L58-L60): Start banner confirms generator phase and run context.
- [data/Data_.py#L194-L195](../data/Data_.py#L194-L195): Summary confirms scale and date range validity.
- [data/Data_.py#L198-L202](../data/Data_.py#L198-L202): Weekday means validate high-level pattern intent.
- [data/Data_.py#L208-L212](../data/Data_.py#L208-L212): Monday vs Wednesday hourly comparison validates differential workload behavior.
- [data/Data_.py#L214-L217](../data/Data_.py#L214-L217): Completion banner provides explicit next action for pipeline continuation.

## Detailed Explanations

### src/Preprocessing (modularized)

- [calendar.py](../src/Preprocessing/calendar.py): Parses the holiday calendar into (month, day) tuples.
- [loader.py](../src/Preprocessing/loader.py): Loads raw CSVs and applies date/holiday/seasonal feature engineering plus data cleaning.
- [features.py](../src/Preprocessing/features.py): Defines the canonical feature order and builds feature DataFrames for training/inference.
- [preprocess.py](../src/Preprocessing/preprocess.py): Thin wrapper that re-exports the above helpers for compatibility.

### src/model_implementation/train_model.py

- [src/model_implementation/train_model.py#L18-L30](../src/model_implementation/train_model.py#L18-L30): Declares canonical input/output paths so all artifacts are written to predictable locations.
- [src/model_implementation/train_model.py#L34-L39](../src/model_implementation/train_model.py#L34-L39): Provides shared metric computation helper reused across train/test/baseline comparisons.
- [src/model_implementation/train_model.py#L42-L44](../src/model_implementation/train_model.py#L42-L44): Creates output directories preemptively to avoid write-time failures.
- [src/model_implementation/train_model.py#L47-L65](../src/model_implementation/train_model.py#L47-L65): Computes data health summary for report transparency.
- [src/model_implementation/train_model.py#L68-L129](../src/model_implementation/train_model.py#L68-L129): Implements plot generators for distribution, heatmap, model comparison, and fit quality.
- [src/model_implementation/train_model.py#L132-L152](../src/model_implementation/train_model.py#L132-L152): Extracts feature importance using native model capability or permutation fallback.
- [src/model_implementation/train_model.py#L155-L240](../src/model_implementation/train_model.py#L155-L240): Evaluates each candidate comprehensively and computes robust selection score.
- [src/model_implementation/train_model.py#L243-L262](../src/model_implementation/train_model.py#L243-L262): Builds chronological split to test temporal generalization.
- [src/model_implementation/train_model.py#L265-L301](../src/model_implementation/train_model.py#L265-L301): Runs fixed sample-case sanity checks on selected model behavior.
- [src/model_implementation/train_model.py#L304-L377](../src/model_implementation/train_model.py#L304-L377): Writes detailed metrics and rationale report for auditability.
- [src/model_implementation/train_model.py#L380-L452](../src/model_implementation/train_model.py#L380-L452): Orchestrates the full training lifecycle and persists chosen model.

### src/Prediction (modularized)

- [context.py](../src/Prediction/context.py): Loads model/data and builds pattern maps for date-aware lookups.
- [patterns.py](../src/Prediction/patterns.py): Pattern-map construction and fallback lookup logic.
- [inference.py](../src/Prediction/inference.py): Feature assembly and prediction routines (deterministic + Monte Carlo).
- [cli.py](../src/Prediction/cli.py): Interactive CLI with weekly/daily/best-time views.
- [predict.py](../src/Prediction/predict.py): Entry point that re-exports the above modules.

### model_zoo and init files

- [src/model_implementation/model_zoo/linear_regression.py#L4-L5](../src/model_implementation/model_zoo/linear_regression.py#L4-L5): Creates linear baseline candidate for reference performance.
- [src/model_implementation/model_zoo/random_forest.py#L4-L15](../src/model_implementation/model_zoo/random_forest.py#L4-L15): Creates random forest candidate with practical defaults and override support.
- [src/model_implementation/model_zoo/gradient_boosting.py#L4-L14](../src/model_implementation/model_zoo/gradient_boosting.py#L4-L14): Creates gradient boosting candidate for residual-focused modeling.
- [src/model_implementation/model_zoo/extra_trees.py#L4-L15](../src/model_implementation/model_zoo/extra_trees.py#L4-L15): Defines optional extra trees candidate for future catalog extension.
- [src/model_implementation/model_zoo/__init__.py#L6-L11](../src/model_implementation/model_zoo/__init__.py#L6-L11): Assembles active model candidates consumed by training benchmark loop.
- [src/model_implementation/__init__.py#L1](../src/model_implementation/__init__.py#L1): Re-exports model catalog API for cleaner package-level imports.

## Recommended Reading Order

- [main.py](../main.py): Start here to understand runtime order and why training precedes prediction.
- [data/Data_.py](../data/Data_.py): Continue here to understand how the dataset itself is generated and validated.
- [src/Preprocessing/preprocess.py](../src/Preprocessing/preprocess.py): Then inspect feature engineering and the exact model input contract.
- [src/model_implementation/train_model.py](../src/model_implementation/train_model.py): Next, study evaluation strategy and model selection logic.
- [src/Prediction/predict.py](../src/Prediction/predict.py): Then inspect inference-time feature assembly and user-facing forecasting behavior.
- [src/model_implementation/model_zoo](../src/model_implementation/model_zoo): Finally, review builder modules to understand candidate configuration options.

## Dependency Handoff Map

- [data/Data_.py#L192](../data/Data_.py#L192): Writes dataset consumed by preprocessing/training and historical-pattern inference flows.
- [src/Preprocessing/loader.py](../src/Preprocessing/loader.py): Returns cleaned feature-ready rows consumed by training pipeline entry.
- [src/model_implementation/model_zoo/__init__.py#L6-L11](../src/model_implementation/model_zoo/__init__.py#L6-L11): Supplies active candidate dictionary consumed by training benchmark loop.
- [src/model_implementation/train_model.py#L434](../src/model_implementation/train_model.py#L434): Persists selected model artifact loaded by prediction module startup.
- [src/model_implementation/train_model.py#L442-L443](../src/model_implementation/train_model.py#L442-L443): Persists benchmark/report artifacts used for analysis and documentation.
- [src/Prediction/cli.py](../src/Prediction/cli.py): Consumes trained model and pattern maps to produce user-facing forecast decisions.

