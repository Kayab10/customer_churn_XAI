"""
Training script — run this once to train the model and save artifacts.

Saves:
  model.json      — XGBoost model in native cross-platform format
  model_meta.pkl  — encoders, feature names, train/test splits (pure Python/numpy)

Usage:
    python train.py
"""

from src.model import load_raw_data, build_pipeline, save_model_artifacts

DATA_PATH = 'data/Churn_Modelling.csv'
MODEL_PATH = 'model.json'
META_PATH  = 'model_meta.pkl'


def main():
    print(f'Loading data from {DATA_PATH}...')
    raw = load_raw_data(DATA_PATH)

    print('Training pipeline...')
    pipeline = build_pipeline(raw)

    print(f'Saving artifacts to {MODEL_PATH} and {META_PATH}...')
    save_model_artifacts(pipeline, MODEL_PATH, META_PATH)

    print('Done. Artifacts saved successfully.')


if __name__ == '__main__':
    main()
