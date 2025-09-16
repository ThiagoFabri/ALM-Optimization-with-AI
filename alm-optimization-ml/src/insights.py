import argparse, json, yaml, joblib, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def weekly_insights(pred_df, k=10):
    txt = []
    high = pred_df.head(k)
    avg_risk = pred_df["risk_score"].mean()
    top_avg = high["risk_score"].mean()
    txt.append(f"Média de risco geral: {avg_risk:.2%} | Top-{k}: {top_avg:.2%}")
    # Modules
    mod = high.groupby("module")["risk_score"].mean().sort_values(ascending=False).head(3)
    if len(mod):
        txt.append("Módulos mais críticos (média de risco): " + ", ".join([f"{m} ({v:.1%})" for m,v in mod.items()]))
    # Environments
    env = high.groupby("environment")["risk_score"].mean().sort_values(ascending=False)
    if len(env):
        txt.append("Ambientes mais críticos: " + ", ".join([f"{m} ({v:.1%})" for m,v in env.items()]))
    return "\n".join(txt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--k", type=int, default=50)
    args = ap.parse_args()
    cfg = load_config(args.config)

    model = joblib.load(cfg["paths"]["model"])
    df = pd.read_csv(cfg["paths"]["data"])
    y = df[cfg["target"]]
    X = df.drop(columns=[cfg["target"], "release_id", "release_datetime"])

    # Permutation importance for explainability
    r = permutation_importance(model, X, y, n_repeats=5, random_state=cfg["seed"])
    imp = pd.DataFrame({"feature": X.columns, "importance": r.importances_mean}).sort_values("importance", ascending=False)
    plt.figure()
    imp.head(15).sort_values("importance").plot(kind="barh", x="feature", y="importance", legend=False)
    plt.title("Feature importance (permutation)")
    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig(cfg["paths"]["importance_plot"])

    # Predictions and text insights
    meta = df[["release_id","release_datetime","module","environment"]].copy()
    proba = model.predict_proba(X)[:,1]
    pred = meta.copy()
    pred["risk_score"] = proba
    pred = pred.sort_values("risk_score", ascending=False)
    pred.to_csv(cfg["paths"]["predictions"], index=False)

    summary = weekly_insights(pred, k=args.k)
    print("== INSIGHTS SEMANAIS ==")
    print(summary)
    print(f"Gráfico salvo em {cfg['paths']['importance_plot']} | Predições em {cfg['paths']['predictions']}")

if __name__ == "__main__":
    main()
