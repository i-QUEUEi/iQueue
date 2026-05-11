# iQueue Code Reference (Line by Line)

This guide is a direct code map for reading the project file-by-file with exact line links.

## File Index

- [main.py](../main.py)
- [data/Data_.py](../data/Data_.py)
- [src/preprocess.py](../src/preprocess.py)
- [src/predict.py](../src/predict.py)
- [src/model_implementation/__init__.py](../src/model_implementation/__init__.py)
- [src/model_implementation/train_model.py](../src/model_implementation/train_model.py)
- [src/model_implementation/model_zoo/__init__.py](../src/model_implementation/model_zoo/__init__.py)
- [src/model_implementation/model_zoo/linear_regression.py](../src/model_implementation/model_zoo/linear_regression.py)
- [src/model_implementation/model_zoo/random_forest.py](../src/model_implementation/model_zoo/random_forest.py)
- [src/model_implementation/model_zoo/gradient_boosting.py](../src/model_implementation/model_zoo/gradient_boosting.py)
- [src/model_implementation/model_zoo/extra_trees.py](../src/model_implementation/model_zoo/extra_trees.py)

## main.py

- Imports: [main.py#L1-L2](../main.py#L1-L2)
- Startup print: [main.py#L5](../main.py#L5)
- Train subprocess call: [main.py#L7](../main.py#L7)
- Completion print: [main.py#L9](../main.py#L9)
- Predict subprocess call: [main.py#L10](../main.py#L10)

## data/Data_.py

### Structure

- Imports and seed: [data/Data_.py#L1-L9](../data/Data_.py#L1-L9)
- Constants and true wait-time patterns: [data/Data_.py#L11-L29](../data/Data_.py#L11-L29)
- Holiday parser function: [data/Data_.py#L31-L54](../data/Data_.py#L31-L54)
- Holiday dates load: [data/Data_.py#L56](../data/Data_.py#L56)
- Startup prints: [data/Data_.py#L58-L60](../data/Data_.py#L58-L60)
- Main generation pipeline: [data/Data_.py#L62-L160](../data/Data_.py#L62-L160)
- DataFrame build and lag features: [data/Data_.py#L162-L170](../data/Data_.py#L162-L170)
- First-row lag defaults loop: [data/Data_.py#L173-L184](../data/Data_.py#L173-L184)
- Fill and weekend feature: [data/Data_.py#L186-L189](../data/Data_.py#L186-L189)
- Save CSV: [data/Data_.py#L192](../data/Data_.py#L192)
- Verification prints and loops: [data/Data_.py#L194-L217](../data/Data_.py#L194-L217)

### All loops

- Holiday text lines: [data/Data_.py#L41](../data/Data_.py#L41)
- Weeks loop: [data/Data_.py#L62](../data/Data_.py#L62)
- Day offsets (Mon-Sat): [data/Data_.py#L67](../data/Data_.py#L67)
- Hours loop: [data/Data_.py#L96](../data/Data_.py#L96)
- Transactions in hour: [data/Data_.py#L110](../data/Data_.py#L110)
- Unique dates for lag defaults: [data/Data_.py#L173](../data/Data_.py#L173)
- Day verification loop: [data/Data_.py#L198](../data/Data_.py#L198)
- Hourly comparison loop: [data/Data_.py#L208](../data/Data_.py#L208)

### All print statements

- Banner prints: [data/Data_.py#L58-L60](../data/Data_.py#L58-L60)
- Transactions generated: [data/Data_.py#L194](../data/Data_.py#L194)
- Date range: [data/Data_.py#L195](../data/Data_.py#L195)
- Verification title: [data/Data_.py#L197](../data/Data_.py#L197)
- Per-day averages print: [data/Data_.py#L202](../data/Data_.py#L202)
- Hourly comparison title and header: [data/Data_.py#L204-L206](../data/Data_.py#L204-L206)
- Per-hour comparison print: [data/Data_.py#L212](../data/Data_.py#L212)
- Final completion prints: [data/Data_.py#L214-L217](../data/Data_.py#L214-L217)

## src/preprocess.py

### Structure

- Imports and holiday path constant: [src/preprocess.py#L1-L7](../src/preprocess.py#L1-L7)
- Holiday month-day parser: [src/preprocess.py#L9-L28](../src/preprocess.py#L9-L28)
- Data loading and feature engineering: [src/preprocess.py#L30-L70](../src/preprocess.py#L30-L70)
- Feature extraction (X, y, features): [src/preprocess.py#L72-L106](../src/preprocess.py#L72-L106)

### All loops

- Holiday calendar lines: [src/preprocess.py#L19](../src/preprocess.py#L19)
- Day distribution print loop: [src/preprocess.py#L66](../src/preprocess.py#L66)

### All print statements

- Loaded record count: [src/preprocess.py#L61](../src/preprocess.py#L61)
- Date range: [src/preprocess.py#L62](../src/preprocess.py#L62)
- Distribution title: [src/preprocess.py#L64](../src/preprocess.py#L64)
- Per-day records print: [src/preprocess.py#L68](../src/preprocess.py#L68)
- Feature stats title and values: [src/preprocess.py#L96-L104](../src/preprocess.py#L96-L104)

## src/predict.py

### Structure

- Imports and paths: [src/predict.py#L1-L14](../src/predict.py#L1-L14)
- Model/data load and startup print: [src/predict.py#L16-L24](../src/predict.py#L16-L24)
- Week/month prep and day names: [src/predict.py#L27-L30](../src/predict.py#L27-L30)
- Holiday month-day parser: [src/predict.py#L32-L51](../src/predict.py#L32-L51)
- Pattern map builder: [src/predict.py#L55-L95](../src/predict.py#L55-L95)
- Pattern maps init: [src/predict.py#L97-L98](../src/predict.py#L97-L98)
- Pattern fallback lookup: [src/predict.py#L100-L117](../src/predict.py#L100-L117)
- Pattern summary prints: [src/predict.py#L119-L123](../src/predict.py#L119-L123)
- Feature list and constants: [src/predict.py#L125-L148](../src/predict.py#L125-L148)
- Feature helper functions: [src/predict.py#L150-L180](../src/predict.py#L150-L180)
- Single-point prediction function: [src/predict.py#L182-L224](../src/predict.py#L182-L224)
- Monte Carlo prediction function: [src/predict.py#L226-L289](../src/predict.py#L226-L289)
- Congestion label function: [src/predict.py#L291-L298](../src/predict.py#L291-L298)
- Weekly forecast renderer: [src/predict.py#L300-L326](../src/predict.py#L300-L326)
- Daily forecast renderer: [src/predict.py#L328-L356](../src/predict.py#L328-L356)
- Best-time finder: [src/predict.py#L358-L390](../src/predict.py#L358-L390)
- Date input parser with loop: [src/predict.py#L392-L408](../src/predict.py#L392-L408)
- CLI menu loop: [src/predict.py#L410-L459](../src/predict.py#L410-L459)

### All loops

- Holiday file lines: [src/predict.py#L42](../src/predict.py#L42)
- Day loop in pattern maps: [src/predict.py#L61](../src/predict.py#L61)
- Hour loop in day-hour map: [src/predict.py#L68](../src/predict.py#L68)
- Week loop in day-week-hour map: [src/predict.py#L71](../src/predict.py#L71)
- Hour loop in week map: [src/predict.py#L74](../src/predict.py#L74)
- Month loop in day-month map: [src/predict.py#L77](../src/predict.py#L77)
- Hour loop in month map: [src/predict.py#L80](../src/predict.py#L80)
- Week loop in day-month-week map: [src/predict.py#L84](../src/predict.py#L84)
- Hour loop in month-week map: [src/predict.py#L87](../src/predict.py#L87)
- Weekly forecast day loop: [src/predict.py#L309](../src/predict.py#L309)
- Weekly forecast hour loop: [src/predict.py#L315](../src/predict.py#L315)
- Daily forecast hour loop: [src/predict.py#L339](../src/predict.py#L339)
- Best-time hour loop: [src/predict.py#L367](../src/predict.py#L367)
- Input retry loop: [src/predict.py#L395](../src/predict.py#L395)
- Main menu loop: [src/predict.py#L422](../src/predict.py#L422)

### All print statements

- Pattern load startup: [src/predict.py#L24](../src/predict.py#L24)
- Pattern summary block: [src/predict.py#L119-L123](../src/predict.py#L119-L123)
- Weekly forecast prints: [src/predict.py#L304-L306](../src/predict.py#L304-L306), [src/predict.py#L311-L313](../src/predict.py#L311-L313), [src/predict.py#L323-L326](../src/predict.py#L323-L326)
- Daily forecast prints: [src/predict.py#L332-L337](../src/predict.py#L332-L337), [src/predict.py#L351-L356](../src/predict.py#L351-L356)
- Best-time prints: [src/predict.py#L362-L364](../src/predict.py#L362-L364), [src/predict.py#L374-L381](../src/predict.py#L374-L381), [src/predict.py#L383-L390](../src/predict.py#L383-L390)
- Invalid date/input prints: [src/predict.py#L403](../src/predict.py#L403), [src/predict.py#L406](../src/predict.py#L406)
- CLI intro and menu prints: [src/predict.py#L411-L420](../src/predict.py#L411-L420), [src/predict.py#L423-L429](../src/predict.py#L423-L429)
- Exit and invalid-choice prints: [src/predict.py#L446-L450](../src/predict.py#L446-L450), [src/predict.py#L454](../src/predict.py#L454)

## src/model_implementation/__init__.py

- Re-export model catalog function: [src/model_implementation/__init__.py#L1](../src/model_implementation/__init__.py#L1)
- Loops: none
- Print statements: none

## src/model_implementation/train_model.py

### Structure

- Imports and matplotlib backend setup: [src/model_implementation/train_model.py#L1-L15](../src/model_implementation/train_model.py#L1-L15)
- Path constants and random state: [src/model_implementation/train_model.py#L18-L32](../src/model_implementation/train_model.py#L18-L32)
- compute_metrics: [src/model_implementation/train_model.py#L34-L39](../src/model_implementation/train_model.py#L34-L39)
- ensure_output_dirs: [src/model_implementation/train_model.py#L42-L44](../src/model_implementation/train_model.py#L42-L44)
- evaluate_data_quality: [src/model_implementation/train_model.py#L47-L65](../src/model_implementation/train_model.py#L47-L65)
- plot_target_distribution: [src/model_implementation/train_model.py#L68-L76](../src/model_implementation/train_model.py#L68-L76)
- plot_day_hour_heatmap: [src/model_implementation/train_model.py#L79-L97](../src/model_implementation/train_model.py#L79-L97)
- plot_model_comparison: [src/model_implementation/train_model.py#L100-L116](../src/model_implementation/train_model.py#L100-L116)
- plot_actual_vs_predicted: [src/model_implementation/train_model.py#L119-L129](../src/model_implementation/train_model.py#L119-L129)
- get_feature_importance: [src/model_implementation/train_model.py#L132-L152](../src/model_implementation/train_model.py#L132-L152)
- evaluate_model: [src/model_implementation/train_model.py#L155-L240](../src/model_implementation/train_model.py#L155-L240)
- chronological_split: [src/model_implementation/train_model.py#L243-L262](../src/model_implementation/train_model.py#L243-L262)
- sample_predictions: [src/model_implementation/train_model.py#L265-L301](../src/model_implementation/train_model.py#L265-L301)
- write_report: [src/model_implementation/train_model.py#L304-L377](../src/model_implementation/train_model.py#L304-L377)
- main training pipeline: [src/model_implementation/train_model.py#L380-L452](../src/model_implementation/train_model.py#L380-L452)

### All loops

- Test-case loop in sample_predictions: [src/model_implementation/train_model.py#L279](../src/model_implementation/train_model.py#L279)
- Model report row loop in write_report: [src/model_implementation/train_model.py#L325](../src/model_implementation/train_model.py#L325)
- Feature importance row loop in write_report: [src/model_implementation/train_model.py#L376](../src/model_implementation/train_model.py#L376)
- Model training/evaluation loop in main: [src/model_implementation/train_model.py#L396](../src/model_implementation/train_model.py#L396)
- Model comparison print loop in main: [src/model_implementation/train_model.py#L422](../src/model_implementation/train_model.py#L422)

### All print statements

- Sample prediction title and rows: [src/model_implementation/train_model.py#L266](../src/model_implementation/train_model.py#L266), [src/model_implementation/train_model.py#L301](../src/model_implementation/train_model.py#L301)
- Model performance heading and rows: [src/model_implementation/train_model.py#L421-L423](../src/model_implementation/train_model.py#L421-L423)
- Selected model and baselines: [src/model_implementation/train_model.py#L429-L432](../src/model_implementation/train_model.py#L429-L432)
- Saved model print: [src/model_implementation/train_model.py#L435](../src/model_implementation/train_model.py#L435)
- Final artifact prints: [src/model_implementation/train_model.py#L447-L448](../src/model_implementation/train_model.py#L447-L448)

## src/model_implementation/model_zoo/__init__.py

- Imports model builders: [src/model_implementation/model_zoo/__init__.py#L1-L3](../src/model_implementation/model_zoo/__init__.py#L1-L3)
- Catalog function: [src/model_implementation/model_zoo/__init__.py#L6-L11](../src/model_implementation/model_zoo/__init__.py#L6-L11)
- Loops: none
- Print statements: none

## src/model_implementation/model_zoo/linear_regression.py

- Import: [src/model_implementation/model_zoo/linear_regression.py#L1](../src/model_implementation/model_zoo/linear_regression.py#L1)
- Builder function: [src/model_implementation/model_zoo/linear_regression.py#L4-L5](../src/model_implementation/model_zoo/linear_regression.py#L4-L5)
- Loops: none
- Print statements: none

## src/model_implementation/model_zoo/random_forest.py

- Import: [src/model_implementation/model_zoo/random_forest.py#L1](../src/model_implementation/model_zoo/random_forest.py#L1)
- Builder function: [src/model_implementation/model_zoo/random_forest.py#L4-L15](../src/model_implementation/model_zoo/random_forest.py#L4-L15)
- Loops: none
- Print statements: none

## src/model_implementation/model_zoo/gradient_boosting.py

- Import: [src/model_implementation/model_zoo/gradient_boosting.py#L1](../src/model_implementation/model_zoo/gradient_boosting.py#L1)
- Builder function: [src/model_implementation/model_zoo/gradient_boosting.py#L4-L14](../src/model_implementation/model_zoo/gradient_boosting.py#L4-L14)
- Loops: none
- Print statements: none

## src/model_implementation/model_zoo/extra_trees.py

- Import: [src/model_implementation/model_zoo/extra_trees.py#L1](../src/model_implementation/model_zoo/extra_trees.py#L1)
- Builder function: [src/model_implementation/model_zoo/extra_trees.py#L4-L15](../src/model_implementation/model_zoo/extra_trees.py#L4-L15)
- Loops: none
- Print statements: none

## Recommended Reading Order

- Start orchestration: [main.py](../main.py)
- Data generation logic: [data/Data_.py](../data/Data_.py)
- Training preprocessing: [src/preprocess.py](../src/preprocess.py)
- Model training and output writing: [src/model_implementation/train_model.py](../src/model_implementation/train_model.py)
- Prediction CLI and interactive loops: [src/predict.py](../src/predict.py)
- Model builder files last: [src/model_implementation/model_zoo](../src/model_implementation/model_zoo)
