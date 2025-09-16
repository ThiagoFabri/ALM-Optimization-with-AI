# ALM Optimization with Machine Learning (GBS)

End-to-end example project (Python) to **analyze and optimize ALM (Application Lifecycle Management)** in a Global Business Services (GBS) context using ML.
It includes:
- Synthetic dataset emulating ALM telemetry (commits, cobertura de testes, bugs, etc.)
- **Modelos de classificação** (Logistic Regression e Gradient Boosting) para prever **risco de falha de deploy** e/ou **incidentes pós-deploy**
- Geração de **insights semanais automatizados** (Agente de IA _rule-based_ + explicabilidade via importância de features)
- **Dashboard Streamlit** para visualização executiva (KPIs, explicabilidade, ranking de risco)
- Pipeline simples de CI (GitHub Actions) + configuração via YAML

> Feito para ser didático e pronto para publicar no GitHub. Ideal para portfólio.

## Visão Geral

- `data/synthetic_release_log.csv`: dados sintéticos (cada linha = release/implantação)
- `src/train.py`: treina e salva o modelo + métricas
- `src/infer.py`: roda inferência e gera ranking de risco
- `src/generate_synthetic.py`: gera/atualiza dados sintéticos
- `src/insights.py`: gera insights semanais (texto + gráfico de importância)
- `app/streamlit_app.py`: dashboard para visualizar KPIs e riscos
- `configs/config.yaml`: hiperparâmetros e caminhos padrão
- `.github/workflows/ci.yml`: lint + testes básicos

## Instalação

```bash
python -m venv .venv && source .venv/bin/activate  # (Linux/Mac)
# ou: .venv\Scripts\activate  (Windows)
pip install -r requirements.txt
```

## Uso Rápido

1) (Opcional) Regere os dados:
```bash
python src/generate_synthetic.py --rows 1500
```

2) Treine o modelo:
```bash
python src/train.py
```

3) Rode inferência para a semana atual:
```bash
python src/infer.py --top_k 50
```

4) Gere insights semanais:
```bash
python src/insights.py
```

5) Rode o dashboard:
```bash
streamlit run app/streamlit_app.py
```

## Ideias de Extensão
- Integrar dados reais do seu pipeline (GitHub/Jira/Jenkins)
- Trocar o modelo por XGBoost e comparar AUC/PR
- Adicionar explicabilidade com SHAP
- Publicar o app no Streamlit Cloud ou Hugging Face Spaces

---

Feito por Thiago Fabri — baseado em experiências com clusterização, lead scoring e automatização de análises semanais.
