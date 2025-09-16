import json, argparse, yaml, joblib, os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(path):
    if not os.path.exists(path):
        # generate default small dataset
        from generate_synthetic import gen_rows
        df = gen_rows(800)
    else:
        df = pd.read_csv(path)
    return df

def build_pipeline(model_type, model_params):
    cat = ["module", "environment"]
    num = ["commits","lines_changed","test_coverage","defects_open","prior_failed_deploys",
           "build_success_rate","lead_time_days","cycle_time_days","code_smells",
           "vulnerabilities","team_experience_years","hour_of_day","day_of_week"]
    pre = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), cat),
        ("scale", StandardScaler(), num)
    ])
    if model_type == "logistic_regression":
        model = LogisticRegression(max_iter=500)
    elif model_type == "xgboost" and HAS_XGB:
        model = XGBClassifier(**model_params)
    else:
        model = GradientBoostingClassifier(**model_params)
    pipe = Pipeline([("pre", pre), ("clf", model)])
    return pipe, cat + num

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    df = load_data(cfg["paths"]["data"])
    y = df[cfg["target"]]
    X = df.drop(columns=[cfg["target"], "release_id", "release_datetime"])

    pipe, feat_names = build_pipeline(cfg["model"]["type"], cfg["model"]["params"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["seed"], stratify=y
    )
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    ap = average_precision_score(y_test, proba)
    report = classification_report(y_test, (proba>0.5).astype(int), output_dict=True)

    os.makedirs("data", exist_ok=True)
    joblib.dump(pipe, cfg["paths"]["model"])
    with open(cfg["paths"]["metrics"], "w") as f:
        json.dump({"roc_auc": auc, "avg_precision": ap, "report": report}, f, indent=2)
    print(f"Saved model to {cfg['paths']['model']} | AUC={auc:.3f} AP={ap:.3f}")

if __name__ == "__main__":
    main()
