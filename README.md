## EN
## Project: Customer Churn Prediction with XGBoost

This project demonstrates, step by step, how to train an XGBoost model to predict customer churn using a reproducible Data Science pipeline.

### Structure

```
predict_churn_XGBoost/
  ├─ src/
  │  ├─ data/
  │  ├─ features/
  │  ├─ models/
  │  ├─ utils/
  │  └─ visualization/
  ├─ notebooks/
  ├─ data/
  │  ├─ raw/
  │  └─ processed/
  ├─ models/
  ├─ reports/
  │  └─ figures/
  ├─ config/
  │  └─ config.yaml
  ├─ .devcontainer/
  │  └─ devcontainer.json
  ├─ requirements.txt
  └─ README.md
```

### How to use

1. Open this directory in VS Code and use “Reopen in Container” (Dev Containers). The container will install dependencies with `uv` using `pyproject.toml`.
2. Open the notebook at `notebooks/01_xgboost_churn.ipynb`.
3. Run the cells in order. If there is no data in `data/raw/`, a synthetic dataset will be generated and saved to `data/processed/`.

### Data

- Place your raw data file (e.g., `churn.csv`) in `data/raw/` with a target column named `churn` (0/1). The notebook attempts to automatically detect numeric and categorical columns.

### Reproducibility

- Parameters and paths can be configured in `config/config.yaml`.
- Dependency management with `uv` via `pyproject.toml`.

### Using uv locally (outside the container)

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate
uv sync
```

### License

Free for educational and portfolio use.




## PT-BR
## Projeto: Predição de Churn com XGBoost

Este projeto demonstra, passo a passo, como treinar um modelo de XGBoost para prever churn de clientes usando um pipeline de Ciência de Dados reproduzível.

### Estrutura

```
predict_churn_XGBoost/
  ├─ src/
  │  ├─ data/
  │  ├─ features/
  │  ├─ models/
  │  ├─ utils/
  │  └─ visualization/
  ├─ notebooks/
  ├─ data/
  │  ├─ raw/
  │  └─ processed/
  ├─ models/
  ├─ reports/
  │  └─ figures/
  ├─ config/
  │  └─ config.yaml
  ├─ .devcontainer/
  │  └─ devcontainer.json
  ├─ requirements.txt
  └─ README.md
```

### Como usar

1. Abra este diretório no VS Code e use "Reopen in Container" (Dev Containers). O container instalará dependências com `uv` usando `pyproject.toml`.
2. Abra o notebook em `notebooks/01_xgboost_churn.ipynb`.
3. Execute as células em ordem. Se não houver dados em `data/raw/`, um dataset sintético será gerado e salvo em `data/processed/`.

### Dados

- Coloque seu arquivo de dados bruto (por exemplo, `churn.csv`) em `data/raw/` com uma coluna alvo chamada `churn` (0/1). O notebook tenta detectar automaticamente colunas numéricas e categóricas.

### Reprodutibilidade

- Parâmetros e caminhos podem ser configurados em `config/config.yaml`.
- Gerenciamento de dependências com `uv` via `pyproject.toml`.

### Usando uv localmente (fora do container)

```
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate
uv sync
```

### Licença

Livre para uso educacional e de portfólio.


