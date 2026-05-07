import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import lime.lime_tabular
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve
from typing import Dict, List, Tuple

FEATURE_COLUMNS = [
    'CreditScore',
    'Geography',
    'Gender',
    'Age',
    'Tenure',
    'Balance',
    'NumOfProducts',
    'HasCrCard',
    'IsActiveMember',
    'EstimatedSalary',
]
CATEGORICAL_FEATURES = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
MODEL_PARAMS = {
    'n_estimators': 300,
    'seed': 101,
    'scale_pos_weight': 4,
    'eval_metric': 'aucpr',
    'use_label_encoder': False,
}


def load_raw_data(path: str = 'data/Churn_Modelling.csv') -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_pipeline(raw: pd.DataFrame):
    data = raw.copy()
    labels = data['Exited'].values

    data = data.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
    categorical_encoders: Dict[str, LabelEncoder] = {}
    categorical_names: Dict[str, np.ndarray] = {}

    for feature in CATEGORICAL_FEATURES:
        encoder = LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature])
        categorical_encoders[feature] = encoder
        categorical_names[feature] = encoder.classes_

    feature_names = data.columns.tolist()
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    class_names = list(label_encoder.classes_)

    train, test, labels_train, labels_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=MODEL_PARAMS['seed'],
    )

    model = xgb.XGBClassifier(**MODEL_PARAMS)
    model.fit(train, labels_train)

    shap_explainer = shap.TreeExplainer(model)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        train.values,
        feature_names=feature_names,
        class_names=class_names,
        categorical_features=[feature_names.index(f) for f in CATEGORICAL_FEATURES],
        categorical_names={f: list(categorical_names[f]) for f in CATEGORICAL_FEATURES},
        kernel_width=3,
    )

    return {
        'raw': raw,
        'data': data,
        'labels': labels,
        'feature_names': feature_names,
        'categorical_features': CATEGORICAL_FEATURES,
        'categorical_encoders': categorical_encoders,
        'categorical_names': categorical_names,
        'class_names': class_names,
        'train': train,
        'test': test,
        'labels_train': labels_train,
        'labels_test': labels_test,
        'model': model,
        'shap_explainer': shap_explainer,
        'lime_explainer': lime_explainer,
    }


def encode_features(record: dict, categorical_encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    record_copy = record.copy()

    for feature in ['HasCrCard', 'IsActiveMember']:
        if isinstance(record_copy[feature], str):
            record_copy[feature] = 1 if record_copy[feature].lower() in ['yes', 'true', '1'] else 0

    for feature, encoder in categorical_encoders.items():
        if feature in record_copy and not isinstance(record_copy[feature], (int, float)):
            record_copy[feature] = encoder.transform([record_copy[feature]])[0]

    return pd.DataFrame([record_copy], columns=FEATURE_COLUMNS)


def predict_customer(model, row: pd.DataFrame) -> Tuple[int, float]:
    proba = model.predict_proba(row)[0][1]
    label = int(proba >= 0.4)  # Lower threshold for imbalanced data
    return label, float(proba)


def evaluate_model(model, test: pd.DataFrame, labels_test: np.ndarray) -> dict:
    y_pred = model.predict(test)
    y_pred_proba = model.predict_proba(test)[:, 1]
    accuracy = accuracy_score(labels_test, y_pred)
    conf_mat = confusion_matrix(labels_test, y_pred)
    report = classification_report(labels_test, y_pred, output_dict=True)
    precision, recall, thresholds = precision_recall_curve(labels_test, y_pred_proba)

    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_mat,
        'classification_report': report,
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
    }


def get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    importance = model.feature_importances_
    return pd.DataFrame(
        {'feature': feature_names, 'importance': importance}
    ).sort_values('importance', ascending=False)


def get_shap_values(shap_explainer, data: pd.DataFrame):
    shap_values = shap_explainer.shap_values(data)
    return shap_values


def explain_customer_lime(
    lime_explainer,
    predict_fn,
    sample: np.ndarray,
    num_features: int = 5,
) -> pd.DataFrame:
    exp = lime_explainer.explain_instance(sample, predict_fn, num_features=num_features)
    return pd.DataFrame(exp.as_list(), columns=['feature', 'contribution'])


def customer_risk_level(probability: float) -> str:
    if probability >= 0.75:
        return 'High'
    if probability >= 0.45:
        return 'Medium'
    return 'Low'


def customer_recommendations(probability: float) -> List[str]:
    if probability >= 0.75:
        return [
            'Offer a loyalty discount or long-term contract',
            'Assign proactive retention support',
            'Monitor account for high churn risk in next 30 days',
        ]
    if probability >= 0.45:
        return [
            'Reach out with personalized service offers',
            'Recommend a service bundle or upgrade',
            'Review billing and payment plan for satisfaction',
        ]
    return [
        'Maintain current relationship and reward loyalty',
        'Highlight stable service and benefits',
        'Keep customers engaged with value offers',
    ]


def load_pipeline_from_pkl(path: str = 'model.pkl'):
    """Load a trained pipeline from a pickle file."""
    import dill
    with open(path, 'rb') as f:
        pipeline = dill.load(f)
    return pipeline
