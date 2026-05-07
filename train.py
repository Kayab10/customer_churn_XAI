"""
Training script — run this once to train the model and save it to model.pkl.

Usage:
    python train.py
"""

import dill
from src.model import load_raw_data, build_pipeline

DATA_PATH = 'data/Churn_Modelling.csv'
MODEL_PATH = 'model.pkl'


def main():
    print(f'Loading data from {DATA_PATH}...')
    raw = load_raw_data(DATA_PATH)

    print('Training pipeline...')
    pipeline = build_pipeline(raw)

    print(f'Saving pipeline to {MODEL_PATH}...')
    with open(MODEL_PATH, 'wb') as f:
        dill.dump(pipeline, f)

    print('Done. Model saved successfully.')


if __name__ == '__main__':
    main()
