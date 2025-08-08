FROM python:3.9-slim as base



WORKDIR /workspace

# Ambiente não interativo e flags do pip
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Dependências de sistema (compilação e runtime) úteis para libs científicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar Astral uv (CLI) e expor no PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -sf /root/.local/bin/uv /usr/local/bin/uv && \
    uv --version


# # Evitar prompts interativos e acelerar installs
# ENV DEBIAN_FRONTEND=noninteractive \
#     PIP_DISABLE_PIP_VERSION_CHECK=1 \
#     PIP_NO_CACHE_DIR=1

# # Dependências do sistema (se necessário para libs científicas)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# WORKDIR /workspace

# O código será montado por volume via docker-compose; deps serão instaladas no postCreate

# Usuário padrão já é vscode na imagem-base

