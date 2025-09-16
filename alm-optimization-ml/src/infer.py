import argparse, json, yaml, joblib, os
import pandas as pd

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--top_k", type=int, default=50)
    args = ap.parse_args()
    cfg = load_config(args.config)

    model = joblib.load(cfg["paths"]["model"])
    df = pd.read_csv(cfg["paths"]["data"])
    meta = df[["release_id","release_datetime","module","environment"]].copy()
    X = df.drop(columns=[cfg["target"], "release_id", "release_datetime"])
    proba = model.predict_proba(X)[:,1]
    out = meta.copy()
    out["risk_score"] = proba
    out = out.sort_values("risk_score", ascending=False)
    out.to_csv(cfg["paths"]["predictions"], index=False)
    print(out.head(args.top_k).to_string(index=False))

if __name__ == "__main__":
    main()
