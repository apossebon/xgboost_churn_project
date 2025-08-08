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


