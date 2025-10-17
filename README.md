# Previsão de Vendas — Projeto de Machine Learning

Este repositório contém um projeto exemplo para treinar, avaliar e implantar um modelo de Machine Learning que prevê vendas futuras a partir de dados históricos. O objetivo é demonstrar um fluxo completo:
- coleta / geração de dados (ex.: dados sintéticos de exemplo);
- preparação e engenharia de features (lags, rolling averages, datas);
- seleção e treino de modelos (Regressão Linear, Random Forest, XGBoost);
- avaliação (MAE, RMSE, R²);
- salvar o modelo e scaler;
- implantação de uma API para previsões em tempo real (FastAPI + Docker).

Estrutura sugerida
- data/
  - generate_synthetic_data.py — script para gerar um CSV de exemplo (train.csv).
- src/
  - preprocess.py — carregamento e transformação dos dados.
  - train.py — script para treinar e salvar o modelo.
  - predict.py — utilitário para carregar modelo e gerar previsões.
  - model.py — definicões de modelos/pipe (opcional).
- app/
  - main.py — API FastAPI para previsões em tempo real.
- models/ — saída: model.joblib, scaler.joblib
- requirements.txt
- Dockerfile
- .github/workflows/ci.yml — exemplo de CI que roda testes básicos.
- tests/ — testes unitários simples.

Como começar (local)
1. Crie e ative um ambiente virtual:
   python -m venv venv
   source venv/bin/activate  # Linux / macOS
   venv\Scripts\activate     # Windows

2. Instale dependências:
   pip install -r requirements.txt

3. Gere dados de exemplo:
   python data/generate_synthetic_data.py --out data/train.csv --n-stores 5 --n-products 20 --periods 730

4. Treine o modelo:
   python src/train.py --data-path data/train.csv --model random_forest --output-dir models

   Isso gera models/model.joblib e models/scaler.joblib.

5. Rode a API localmente:
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   - Endpoint: POST /predict
     -> Recebe JSON com linhas (mesmas colunas de entrada) e retorna previsões.

Deployment (Docker)
- Construa a imagem:
  docker build -t sales-forecast:latest .

- Rode:
  docker run -p 8000:8000 sales-forecast:latest

Boas práticas e próximos passos
- Validar por loja/produto de forma separada.
- Tunar hiperparâmetros (GridSearch / Optuna).
- Implementar time-series cross-validation (ex.: expanding window).
- Adotar monitoramento de deriva de dados e re-treinamento automático.
- Persistir modelos com versionamento (MLflow, DVC, S3).
- Teste com dados reais e ajuste das features.

Se quiser, adapto o pipeline:
- adicionar modelos (Prophet, LSTM, LightGBM);
- ajustar features de calendário (feriados);
- integrar com um serviço cloud (SageMaker, Vertex AI, Azure ML);
- criar notebooks de análise (exploração/visualização).

Autor: gerado por abghajkb24 (pedido).
