# FII Predictor — Predição de Dividend Yield com Machine Learning

> **Trabalho de Conclusão de Curso (TCC)**  
> Curso: Gestão da Tecnologia da Informação — IFSC  
> Orientador: Prof. Egon Sewald Junior  
> Tema: ANÁLISE PREDITIVA PARA RENTABILIDADE DE FUNDOS DE INVESTIMENTO IMOBILIÁRIO (FII) DESTINADA A INVESTIDORES SEM CONHECIMENTO EM PROGRAMAÇÃO

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Arquitetura da Solução](#2-arquitetura-da-solução)
3. [Decisões Metodológicas](#3-decisões-metodológicas)
4. [Fontes de Dados](#4-fontes-de-dados)
5. [Coleta e Montagem do Dataset](#5-coleta-e-montagem-do-dataset)
6. [Pré-processamento e Feature Engineering](#6-pré-processamento-e-feature-engineering)
7. [Treinamento dos Modelos](#7-treinamento-dos-modelos)
8. [Resultados dos Modelos](#8-resultados-dos-modelos)
9. [API de Machine Learning](#9-api-de-machine-learning)
10. [Interface Web](#10-interface-web)
11. [Deploy e Infraestrutura](#11-deploy-e-infraestrutura)
12. [Segmentos Excluídos e Justificativas](#12-segmentos-excluídos-e-justificativas)
13. [Limitações da Pesquisa](#13-limitações-da-pesquisa)
14. [Como Replicar](#14-como-replicar)
15. [Estrutura do Repositório](#15-estrutura-do-repositório)
16. [Referências](#16-referências)

---

## 1. Visão Geral

Esta pesquisa aplicada desenvolveu uma solução completa de análise preditiva do **Dividend Yield (DY) mensal** de FIIs brasileiros, composta por:

- **Pipeline de dados** em Python para coleta e estruturação via brapi.dev Pro e BCB SGS
- **Modelos de Machine Learning** treinados por segmento (Random Forest)
- **API REST** (FastAPI) com predição recursiva de até 12 meses
- **Interface web** (HTML/JS) hospedada no Vercel, sem necessidade de conhecimento técnico pelo usuário

O objetivo central é permitir que investidores pessoa física avaliem tendências de DY de FIIs de forma acessível, com validação baseada em dados reais de 2025.

---

## 2. Arquitetura da Solução

```
┌─────────────────────────────────────────────────────────┐
│                    COLETA DE DADOS                      │
│   brapi.dev Pro (DY, P/VP)  +  BCB SGS 4390 (SELIC)   │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│              DATASET (2019–2024)                        │
│   dataset_fiis_2019_2024_brapi_v2.xlsx                 │
│   6.871 linhas · 44 fundos · 4 segmentos               │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│           TREINAMENTO (treinar_modelo_v4.py)            │
│   Random Forest por segmento · com/sem pandemia        │
│   Features: DY_lag1, DY_lag2, DY_lag3, PVP_lag1, SELIC│
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│              MODELOS (.pkl) + meta.json                 │
│   modelo_logistico.pkl  · modelo_hibrido.pkl           │
│   modelo_escritorios.pkl · modelo_shoppings.pkl        │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│         API REST FastAPI (api.py) — Render.com          │
│   POST /predict/serie → predição recursiva 12 meses    │
│   GET  /health        → métricas dos modelos           │
│   GET  /fundos        → lags reais por fundo           │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│          INTERFACE WEB (index.html) — Vercel           │
│   Seleção de fundo → previsão → 4 gráficos interativos │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Decisões Metodológicas

### 3.1 Estratégia de Modelagem

Após iterações com diferentes abordagens, adotou-se **um modelo por segmento**:

| Abordagem | R² médio | Problema |
|---|---|---|
| Modelo global único | < 0 | Alta heterogeneidade entre fundos |
| Z-score global | ~0.13 | MAPE instável |
| Por segmento (versão final) | **0.1667** | Melhor equilíbrio viés-variância |

### 3.2 Algoritmo Escolhido

**Random Forest** foi o algoritmo final, aplicado uniformemente a todos os segmentos:

- Robusto a outliers — relevante dado o período pandêmico (2020)
- Não requer normalização das features
- Permite inspeção de importância de variáveis
- Interpretável para fins acadêmicos

> BREIMAN, L. Random Forests. *Machine Learning*, v.45, n.1, p.5–32, 2001.

### 3.3 Predição Recursiva

Para gerar séries de até 12 meses, utilizou-se **predição recursiva**:

```
lag1=dez/2024, lag2=nov/2024, lag3=out/2024
→ predict() → dy_jan/2025  (mês 1)
→ predict(dy_jan, dez, nov) → dy_fev/2025  (mês 2)
→ predict(dy_fev, dy_jan, dez) → dy_mar/2025  (mês 3)
...
```

Os lags iniciais são os 3 últimos valores reais de 2024 extraídos da brapi.dev Pro antes do cancelamento da assinatura.

### 3.4 Tratamento da Pandemia

Dois modelos foram treinados por segmento:
- **Com pandemia** (2020-03 a 2020-12 incluídos)
- **Sem pandemia** (período excluído)

O período pandêmico causou distribuições atípicas — especialmente em Shoppings, cujos DYs zeraram durante o fechamento do comércio. A comparação entre as versões está disponível na interface.

### 3.5 Shoppings — Normalização Z-Score

O segmento Shoppings possui alta dispersão de DY entre os 15 fundos (escalas muito diferentes). A normalização Z-score por fundo foi aplicada antes do treino e revertida na predição:

```python
dy_normalizado = (dy - media_fundo) / std_fundo
dy_desnormalizado = dy_previsto * std_fundo + media_fundo
```

---

## 4. Fontes de Dados

| Fonte | Dado | Série/Endpoint | Período |
|---|---|---|---|
| brapi.dev Pro | DY mensal (dividendYield1m) | `/api/v2/fii/indicators/history` | 2019–2025 |
| brapi.dev Pro | P/VP (priceToNav) | `/api/v2/fii/indicators/history` | 2019–2025 |
| brapi.dev Pro | Lista de FIIs | `/api/v2/fii/list` | — |
| BCB SGS | SELIC mensal | Série 4390 | 2019–2024 |
| investidor10.com.br | Segmento dos FIIs Multicategoria | Manual | — |

> **Nota:** A assinatura brapi.dev Pro foi cancelada após extração completa dos dados. Todos os dados históricos (2019–2025) foram preservados nos arquivos do repositório.

---

## 5. Coleta e Montagem do Dataset

### 5.1 Script principal

```bash
python montar_dataset_brapi.py
```

O script `montar_dataset_brapi.py` realiza:
1. Busca a lista completa de FIIs via `/api/v2/fii/list`
2. Para cada lote de até 20 fundos, busca o histórico mensal de DY e P/VP
3. Monta um DataFrame consolidado e exporta para `.xlsx`

**Autenticação:** Bearer token via variável de ambiente `BRAPI_TOKEN` no arquivo `.env`.

### 5.2 Correção de Segmentos (Multicategoria)

Fundos classificados como "Multicategoria" pela brapi foram remapeados manualmente para seus segmentos corretos via `corrigir_dataset.py`, usando como referência o site investidor10.com.br.

```bash
python corrigir_dataset.py
```

Gera: `dataset_fiis_2019_2024_brapi_v2.xlsx`

### 5.3 Extração de dados 2024–2025 (antes do cancelamento)

```bash
python extrair_tudo.py
```

Gera: `dados_fiis_2024_2025.json` — 65 fundos × 2 anos, com DY e P/VP mensais.

Os dados de 2025 foram convertidos para arrays JavaScript e embutidos diretamente no `index.html` via:

```bash
python gerar_dados_html.py > dados_para_html.txt
```

Isso eliminou a dependência da brapi.dev em tempo de execução da interface.

---

## 6. Pré-processamento e Feature Engineering

### 6.1 Features utilizadas

| Feature | Descrição | Fonte |
|---|---|---|
| `DY_lag1` | DY do mês anterior | Dataset |
| `DY_lag2` | DY de 2 meses atrás | Dataset |
| `DY_lag3` | DY de 3 meses atrás | Dataset |
| `PVP_lag1` | P/VP do mês anterior | Dataset |
| `SELIC` | Taxa SELIC mensal | BCB SGS 4390 |
| `Tipo_do_Fundo` | Tijolo ou Papel (one-hot) | Dataset |

> `Tipo_do_Fundo` é incluído apenas nos segmentos com mais de 1 tipo (ex: Híbrido).

### 6.2 Variável alvo

```python
DY_target = DY do mês seguinte (shift(-1))
```

### 6.3 Divisão temporal

Para respeitar a natureza temporal dos dados (evitar *data leakage*):

```python
data_corte = quantile(0.8) dos dados cronológicos
treino: dados anteriores à data de corte (~80%)
teste:  dados posteriores (~20%)
```

### 6.4 Validação cruzada temporal

`TimeSeriesSplit` com 5 folds — cada fold treina em um período maior e testa no período seguinte, respeitando a ordem cronológica.

---

## 7. Treinamento dos Modelos

### 7.1 Execução

```bash
python treinar_modelo_v4.py --arquivo dataset_fiis_2019_2024_brapi_v2.xlsx
```

### 7.2 Hiperparâmetros do Random Forest

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=3,
    max_features=0.7,
    random_state=42,
    n_jobs=-1
)
```

### 7.3 Segmentos e fundos treinados

| Segmento | Fundos | Amostras | Features |
|---|---|---|---|
| Logístico | 10 | 642 | DY_lag1/2/3, PVP_lag1, SELIC |
| Híbrido | 8 | 519 | DY_lag1/2/3, PVP_lag1, SELIC, Tipo_do_Fundo |
| Escritórios | 11 | 730 | DY_lag1/2/3, PVP_lag1, SELIC |
| Shoppings | 15 | 1.015 | DY_lag1/2/3, PVP_lag1, SELIC (+ z-score) |

### 7.4 Artefatos gerados

```
modelo/
├── modelo_logistico.pkl
├── modelo_logistico_sem_pandemia.pkl
├── modelo_hibrido.pkl
├── modelo_hibrido_sem_pandemia.pkl
├── modelo_escritorios.pkl
├── modelo_escritorios_sem_pandemia.pkl
├── modelo_shoppings.pkl
├── modelo_shoppings_sem_pandemia.pkl
├── random_forest.pkl          ← fallback geral
├── meta.json                  ← métricas e configurações
└── fundos_recentes.csv        ← últimos lags conhecidos por fundo
```

---

## 8. Resultados dos Modelos

### 8.1 R² por segmento

| Segmento | R² com pandemia | R² sem pandemia | Variação |
|---|---|---|---|
| **Logístico** | **0.3693** | 0.3028 | ↓ -0.0665 |
| **Híbrido** | **0.1801** | 0.1459 | ↓ -0.0342 |
| **Escritórios** | **0.1231** | 0.1072 | ↓ -0.0159 |
| **Shoppings** | -0.0056 | -0.0063 | ↓ -0.0007 |
| **Média** | **0.1667** | 0.1374 | — |

> O R² negativo de Shoppings reflete a alta volatilidade do segmento e o impacto persistente da pandemia nos padrões de distribuição.

### 8.2 Importância das variáveis (Random Forest)

| Feature | Importância média |
|---|---|
| DY_lag1 | ~34% |
| Segmento | ~16% |
| DY_lag2 | ~22% |
| P/VP | ~14% |
| DY_lag3 | ~14% |

### 8.3 Validação com dados reais de 2025

O gráfico "Real 2025 vs Previsão do modelo" na interface compara a predição recursiva com os dados reais extraídos da brapi.dev Pro (jan–dez/2025) antes do cancelamento da assinatura.

---

## 9. API de Machine Learning

### 9.1 Endpoints principais

| Método | Endpoint | Descrição |
|---|---|---|
| `GET` | `/` | Status e segmentos disponíveis |
| `GET` | `/health` | Métricas R², MAE, MAPE por segmento |
| `GET` | `/fundos` | Últimos lags reais por fundo |
| `GET` | `/segmentos` | Segmentos disponíveis e excluídos |
| `POST` | `/predict` | Previsão de 1 mês |
| `POST` | `/predict/serie` | Predição recursiva de N meses |
| `POST` | `/predict/comparar` | Comparação com/sem pandemia |

### 9.2 Exemplo de chamada — predição recursiva

```bash
curl -X POST https://fii-predictor.onrender.com/predict/serie \
  -H "Content-Type: application/json" \
  -d '{
    "sigla": "HGLG11",
    "segmento": "Logistico",
    "dy_lag1": 0.00704,
    "dy_lag2": 0.007013,
    "dy_lag3": 0.006989,
    "pvp": 0.8054,
    "modelo": "Random Forest",
    "n_meses": 12
  }'
```

### 9.3 Resposta

```json
{
  "sigla": "HGLG11",
  "segmento": "Logistico",
  "n_meses": 12,
  "serie": [
    {"mes": "2025-01", "dy_previsto": 0.006821, "dy_previsto_pct": 0.6821, "dy_previsto_aa": 8.19},
    {"mes": "2025-02", "dy_previsto": 0.006798, ...},
    ...
  ],
  "r2": 0.3693,
  "mape": 16.6
}
```

### 9.4 Dependências

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
scikit-learn==1.5.1
pandas==2.2.2
numpy==1.26.4
joblib==1.4.2
httpx==0.27.0
python-dotenv==1.0.1
```

---

## 10. Interface Web

A interface (`index.html`) é uma aplicação HTML/CSS/JavaScript pura, sem framework, hospedada no Vercel.

### 10.1 Funcionalidades

- **Seleção de fundo** por segmento e sigla, com DY e P/VP preenchidos automaticamente
- **Predição recursiva** de 3, 6 ou 12 meses via API
- **4 gráficos interativos:**
  1. DY histórico (2024) + previsão (2025)
  2. Real 2025 (brapi) vs Previsão do modelo — validação
  3. Impacto da pandemia (modelo com vs sem 2020)
  4. Desempenho do segmento — DY real 2025 de todos os fundos, com filtro por sigla
- **Alerta** quando o fundo selecionado não esteve no dataset de treino
- **Dados embutidos** — sem dependência de APIs externas em tempo de execução

### 10.2 Dados embutidos no HTML

Para eliminar a dependência da brapi.dev após o cancelamento da assinatura:

| Variável JS | Conteúdo | Fundos |
|---|---|---|
| `REAL_2025` | DY mensal real jan–dez/2025 | 42 fundos |
| `REAL_2024` | DY mensal real jan–dez/2024 | 65 fundos |
| `PVP_2024` | P/VP mensal real jan–dez/2024 | 55 fundos |
| `FUNDOS_TREINO` | Set com siglas do dataset | 44 fundos |

---

## 11. Deploy e Infraestrutura

| Componente | Plataforma | URL |
|---|---|---|
| API ML | Render.com (free tier) | `https://fii-predictor.onrender.com` |
| Interface | Vercel | Configurado via GitHub |
| Código | GitHub Codespaces | Desenvolvimento |

### 11.1 Variáveis de ambiente (Render)

```
BRAPI_TOKEN=<token_brapi>   # necessário apenas para endpoints /fii/{sigla}
```

> Os modelos de predição não dependem da brapi em produção — usam apenas os `.pkl` treinados.

### 11.2 Processo de deploy

```bash
git add .
git commit -m "mensagem"
git push
```

O Render detecta o push automaticamente e reinicia o serviço. O Vercel faz o mesmo para a interface.

---

## 12. Segmentos Excluídos e Justificativas

| Segmento | Motivo |
|---|---|
| **Títulos e Val. Mob.** | DY indexado ao spread individual dos CRIs — variável contratual não disponível em fonte pública. R² negativo confirmado empiricamente mesmo com CDI como feature. |
| **FOF (Fundo de Fundos)** | DY dependente da carteira de outros FIIs e decisões discricionárias do gestor — não capturável com variáveis macroeconômicas públicas. |
| **Hospital** | Apenas 3 fundos com comportamento muito distinto entre si — amostra insuficiente para generalização. |
| **Varejo** | Apenas 3 fundos — amostra insuficiente para treinamento confiável. |
| **Outros** | Grupo heterogêneo sem critério de homogeneidade — mistura de tipos incompatíveis. |

---

## 13. Limitações da Pesquisa

1. **Dados históricos limitados a 2019–2024** — período de 6 anos inclui evento atípico (pandemia COVID-19)
2. **P/VP como proxy de valor** — não captura variações intraday ou eventos corporativos
3. **SELIC como única variável macro** — não inclui IPCA, spread de crédito ou vacância física
4. **Predição recursiva acumula erro** — o erro do mês 1 propaga para os meses seguintes
5. **Shoppings com R² negativo** — alta volatilidade do segmento não capturada pelo modelo
6. **Fundos fora do dataset** — modelo usa fallback geral com menor precisão para fundos não treinados
7. **Dados de 2025 estáticos** — não há atualização automática após cancelamento da brapi

---

## 14. Como Replicar

### Pré-requisitos

```bash
Python 3.11+
pip install -r requirements.txt
```

### Passo 1 — Clonar o repositório

```bash
git clone https://github.com/seu-usuario/fii-predictor.git
cd fii-predictor
```

### Passo 2 — Configurar variáveis de ambiente

```bash
cp .env.example .env
# Editar .env e adicionar BRAPI_TOKEN (necessário apenas para coleta de dados)
```

### Passo 3 — Instalar dependências

```bash
pip install -r requirements.txt
```

### Passo 4 — Treinar os modelos

> Necessário ter o dataset `dataset_fiis_2019_2024_brapi_v2.xlsx` na raiz do projeto.

```bash
python treinar_modelo_v4.py --arquivo dataset_fiis_2019_2024_brapi_v2.xlsx
```

Os modelos serão salvos em `modelo/`.

### Passo 5 — Rodar a API localmente

```bash
uvicorn api:app --reload --port 8000
```

Acesse: `http://localhost:8000/docs`

### Passo 6 — Abrir a interface

Abra o arquivo `index.html` no navegador, ou altere a variável `API` no início do script para apontar para `http://localhost:8000`.

---

## 15. Estrutura do Repositório

```
fii-predictor/
│
├── api.py                          # API FastAPI — endpoints de predição
├── treinar_modelo_v4.py            # Script de treinamento por segmento
├── montar_dataset_brapi.py         # Coleta de dados da brapi.dev Pro
├── corrigir_dataset.py             # Remapeamento de segmentos Multicategoria
├── index.html                      # Interface web (sem dependências externas)
├── requirements.txt                # Dependências Python
├── runtime.txt                     # Versão do Python para o Render
├── .env                            # Variáveis de ambiente (não versionado)
│
├── modelo/
│   ├── modelo_logistico.pkl
│   ├── modelo_logistico_sem_pandemia.pkl
│   ├── modelo_hibrido.pkl
│   ├── modelo_hibrido_sem_pandemia.pkl
│   ├── modelo_escritorios.pkl
│   ├── modelo_escritorios_sem_pandemia.pkl
│   ├── modelo_shoppings.pkl
│   ├── modelo_shoppings_sem_pandemia.pkl
│   ├── random_forest.pkl           # Fallback geral
│   ├── meta.json                   # Métricas e configurações dos modelos
│   └── fundos_recentes.csv         # Últimos lags reais por fundo
│
└── dados_fiis_2024_2025.json       # Dados históricos extraídos da brapi
```

---

## 16. Referências

**Algoritmos de Machine Learning**

- BREIMAN, L. Random Forests. *Machine Learning*, v.45, n.1, p.5–32, 2001. DOI: 10.1023/A:1010933404324

- FRIEDMAN, J.H. Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, v.29, n.5, p.1189–1232, 2001. DOI: 10.1214/aos/1013203451

- HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. *The Elements of Statistical Learning*. 2. ed. New York: Springer, 2009.

**Aplicação em REITs e FIIs**

- LEOW, M. et al. Enhancing real estate investment trust return forecasts using machine learning. *Real Estate Economics*, 2025. DOI: 10.1111/1540-6229.12527

**Fontes de Dados**

- BANCO CENTRAL DO BRASIL. Sistema Gerenciador de Séries Temporais (SGS). Série 4390 — Taxa SELIC. Disponível em: https://www.bcb.gov.br/estatisticas/tabelaespecial

- BRAPI. Documentação da API de dados financeiros brasileiros. Disponível em: https://brapi.dev/docs

---

## Licença

Este projeto foi desenvolvido exclusivamente para fins acadêmicos como Trabalho de Conclusão de Curso (TCC). Os dados utilizados são de fontes públicas ou foram adquiridos via assinatura da plataforma brapi.dev Pro.

**As previsões geradas por esta ferramenta não constituem recomendação de investimento.**

---

*Desenvolvido com Python, FastAPI, scikit-learn, Chart.js · IFSC 2024–2025*
