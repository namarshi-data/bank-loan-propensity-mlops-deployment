# Source Code Package

This `src/` package contains reusable production-style code extracted from the project notebooks.

## Files

| File | Purpose |
|------|---------|
| `config.py` | Central project paths, feature lists, and constants |
| `data_preprocessing.py` | Load, merge, clean, and validate raw datasets |
| `data_preparation.py` | Prepare features and create train-test datasets |
| `evaluate_model.py` | Evaluate models and calculate FP/FN metrics |
| `train_model.py` | Train, tune, evaluate, and save model artifacts |
| `inference.py` | Load saved model and generate predictions |
| `utils.py` | Helper utilities |

## Execution Order

Run from the project root:

```bash
python -m src.data_preprocessing
python -m src.data_preparation
python -m src.train_model
python -m src.inference