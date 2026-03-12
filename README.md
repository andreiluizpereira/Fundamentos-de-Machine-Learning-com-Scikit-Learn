# Projeto da Disciplina — Machine Learning Supervisionado

**Disciplina**: Machine Learning com Scikit-Learn e MLOps  
**Aluno**: Andrei Luiz Pereira  
**Professor**: Icaro Augusto Maccari Zelioli  
**Dataset**: Adult Census Income (UCI Machine Learning Repository)

---

## Sumário

1. [Descrição do Problema e dos Dados](#1-descrição-do-problema-e-dos-dados)
   - 1.1 [Contexto e Motivação](#11-contexto-e-motivação)
   - 1.2 [Desafios do Domínio](#12-desafios-do-domínio)
   - 1.3 [Descrição Técnica do Dataset](#13-descrição-técnica-do-dataset)
   - 1.4 [Variável-Alvo](#14-variável-alvo)
   - 1.5 [Remoção de Features](#15-remoção-de-features)
   - 1.6 [Features Utilizadas](#16-features-utilizadas)

2. [Pré-processamento dos dados](#2-pré-processamento-dos-dados)
   - 2.1 [Remover colunas indesejáveis](#21-remover-colunas-indesejáveis)
   - 2.2 [Separação Teste x Treino](#22-separação-teste-x-treino)
   - 2.3 [Análise de outliers](#23-análise-de-outliers)
   - 2.4 [Análise de nulos](#24-análise-de-nulos)
   - 2.5 [Pipeline de pré-processamento](#25-pipeline-de-pré-processamento)

3. [Modelo Baseline: Classificador Linear com Perceptron](#3-modelo-baseline-classificador-linear-com-perceptron)
   - 3.1 [Treinamento Perceptron](#31-treinamento-perceptron)
   - 3.2 [Previsões e desempenho do Perceptron](#32-previsões-e-desempenho-do-perceptron)
   - 3.3 [Interpretação dos Coeficientes](#33-interpretação-dos-coeficientes)
   - 3.4 [Limitações do Perceptron](#34-limitações-do-perceptron)

4. [Modelo com Árvore de Decisão](#4-modelo-com-árvore-de-decisão)
   - 4.1 [Treinamento Decision Tree](#41-treinamento-decision-tree)
   - 4.2 [Previsões e desempenho do Decision Tree](#42-previsões-e-desempenho-do-decision-tree)
   - 4.3 [Interpretação e Risco de Overfitting](#43-interpretação-e-risco-de-overfitting)
   - 4.4 [Comparação: Perceptron vs. Decision Tree](#44-comparação-perceptron-vs-decision-tree)

5. [Validação Cruzada e Busca de Hiperparâmetros](#5-validação-cruzada-e-busca-de-hiperparâmetros)
   - 5.1 [Definir configurações para otimização](#51-definir-configurações-para-otimização)
   - 5.2 [Executar RandomizedSearchCV](#52-executar-randomizedsearchcv)
   - 5.3 [Melhores hiperparâmetros](#53-melhores-hiperparâmetros)
   - 5.4 [Melhor desempenho médio](#54-melhor-desempenho-médio)
   - 5.5 [Análise de F1-macro em cada fold do melhor modelo](#55-análise-de-f1-macro-em-cada-fold-do-melhor-modelo)
   - 5.6 [Desempenho final do RandomizedSearchCV + StratifiedKFold](#56-desempenho-final-do-randomizedsearchcv--stratifiedkfold)
   - 5.7 [Comparação: Árvore Padrão vs. Árvore Otimizada](#57-comparação-árvore-padrão-vs-árvore-otimizada)
   - 5.8 [Análise dos Resultados](#58-análise-dos-resultados)

6. [Modelo Avançado: SVM Linear](#6-modelo-avançado-svm-linear)
   - 6.1 [Definir configurações para otimização](#61-definir-configurações-para-otimização)
   - 6.2 [Executar SVM Linear](#62-executar-svm-linear)
   - 6.3 [Melhores parâmetros](#63-melhores-parâmetros)
   - 6.4 [Melhor desempenho médio](#64-melhor-desempenho-médio)
   - 6.5 [Desempenho final do SVM Linear](#65-desempenho-final-do-svm-linear)

7. [Comparação Final entre Modelos](#7-comparação-final-entre-modelos)
   - 7.1 [Gráfico de comparação de modelos](#71-gráfico-de-comparação-de-modelos)
   - 7.2 [Discussão](#72-discussão)

---

## Setup do Ambiente

### Criar Ambiente Virtual

#### Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Instalar Dependências

Com o ambiente virtual ativado, execute:

```bash
pip install -r requirements.txt
```

---

## Executar o Projeto

Para rodar os experimentos:

```bash
jupyter notebook projeto_disciplina.ipynb
```

---

## Estrutura de Arquivos

```
.
├── README.md                          # Este arquivo
├── requirements.txt                   # Dependências do projeto
├── projeto_disciplina.ipynb           # Notebook principal
├── data/
│   └── train.csv                      # Dataset Adult Census Income
```

---

## Requisitos

- Python 3.9+
- Veja `requirements.txt` para versões específicas das bibliotecas

---

## Autor

Andrei Luiz Pereira
