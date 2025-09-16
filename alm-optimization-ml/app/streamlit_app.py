import streamlit as st
import pandas as pd
import joblib, yaml
import matplotlib.pyplot as plt

st.set_page_config(page_title="ALM Risk Dashboard", layout="wide")

@st.cache_data
def load_cfg():
    with open("configs/config.yaml","r") as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

cfg = load_cfg()
st.title("ALM Optimization - Risk & Insights")
st.caption("Demo de portfólio • Predição de risco de falha de deploy e insights semanais")

# Side: config
st.sidebar.header("Config")
model_path = cfg["paths"]["model"]
data_path = cfg["paths"]["data"]
pred_path = cfg["paths"]["predictions"]
importance_plot = cfg["paths"]["importance_plot"]

# Data
df = pd.read_csv(data_path)
model = load_model(model_path)

# Predictions
features = df.drop(columns=[cfg["target"], "release_id", "release_datetime"])
meta = df[["release_id","release_datetime","module","environment"]].copy()
proba = model.predict_proba(features)[:,1]
pred = meta.copy()
pred["risk_score"] = proba
pred = pred.sort_values("risk_score", ascending=False)

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Releases", len(df))
col2.metric("Risco médio", f"{pred['risk_score'].mean():.1%}")
col3.metric("Top-10 risco médio", f"{pred.head(10)['risk_score'].mean():.1%}")

st.subheader("Ranking de Risco (Top-50)")
st.dataframe(pred.head(50), use_container_width=True)

st.subheader("Distribuição de Risco")
fig, ax = plt.subplots()
ax.hist(pred["risk_score"], bins=30)
st.pyplot(fig)

st.subheader("Importância de Features (Permutation)")
try:
    import numpy as np
    from sklearn.inspection import permutation_importance
    target = df[cfg["target"]]
    r = permutation_importance(model, features, target, n_repeats=3, random_state=cfg["seed"])
    imp = pd.DataFrame({"feature": features.columns, "importance": r.importances_mean}).sort_values("importance", ascending=False).head(12)
    st.bar_chart(imp.set_index("feature"))
except Exception as e:
    st.info(f"Não foi possível calcular importância: {e}")

st.markdown("---")
st.caption("Feito por Thiago Fabri • Projeto educacional (dados sintéticos)")
