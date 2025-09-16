# ALM-Optimization-with-AI
Projeto de otimização do Application Lifecycle Management (ALM) com Machine Learning e IA, prevendo risco de falhas em deploys e gerando insights automáticos semanais com dashboard interativo.


Este projeto simula um pipeline de ALM corporativo (Application Lifecycle Management) no contexto de Global Business Services (GBS).
O objetivo é mostrar como aplicar ciência de dados e machine learning para aumentar a eficiência e reduzir riscos em ambientes de TI críticos, como bancos e grandes empresas.

# A solução abrange:

Geração de dados sintéticos realistas sobre deploys (commits, bugs, cobertura de testes, vulnerabilidades etc.).

Modelos de ML supervisionados (Logistic Regression, Gradient Boosting, XGBoost) para prever falhas em deploys.

Dashboard em Streamlit para acompanhamento executivo, com KPIs, ranking de risco e importância de variáveis.

Este repositório foi estruturado como um portfólio para demonstrar habilidades em:

Modelagem estatística e preditiva.

Engenharia de dados e automação de pipelines.

Visualização de dados e storytelling analítico.

Aplicação prática de IA em cenários de negócio (ex: operações bancárias e corporativas).

# 🧩 Estrutura
alm-optimization-ml/
│── app/                 # Dashboard em Streamlit
│── configs/             # Arquivos de configuração (YAML)
│── data/                # Dados sintéticos e saídas (modelos, métricas, gráficos)
│── src/                 # Código-fonte (geração, treino, inferência, insights)
│── .github/workflows/   # CI simples com GitHub Actions
│── README.md            # Este documento
│── requirements.txt     # Dependências do projeto


Principais scripts:

generate_synthetic.py → gera dataset sintético de deploys.

train.py → treina e salva modelos de risco.

infer.py → gera ranking de releases com maior probabilidade de falha.

insights.py → cria insights semanais automáticos + gráfico de importância de features.

streamlit_app.py → dashboard interativo.

# ⚙️ Instalação

Crie e ative um ambiente virtual:

py -m venv .venv
.\.venv\Scripts\activate   # Windows

Instale as dependências:

pip install -r requirements.txt

# 🚀 Uso

Gerar dados sintéticos

python src/generate_synthetic.py --rows 1000


Treinar modelo

python src/train.py


Inferência (ranking de risco)

python src/infer.py --top_k 20


Dados Gerais

python src/insights.py


Dashboard interativo

streamlit run app/streamlit_app.py

# 📊 Exemplo de saída

Métricas: ROC-AUC, Average Precision, Classification Report.

Ranking de risco: releases mais críticos por módulo/ambiente.

Insights semanais: texto resumido para gestores.

Dashboard: KPIs, histograma de risco, top-50 releases, importância de variáveis.

# 🌟 Extensões possíveis

Conectar com dados reais de GitHub Actions, Jenkins ou Jira.

Adicionar explicabilidade avançada com SHAP.

Criar módulo de clusterização de releases via PCA + KMeans (mapa 2D).

Publicar no Streamlit Cloud ou Hugging Face Spaces.

# 👤 Autor

Thiago Fabri de Oliveira
📧 thiagofabridoliveira@gmail.com

🔗 LinkedIn
https://www.linkedin.com/in/thiago-fabri/
🔗 GitHub
https://github.com/ThiagoFabri
