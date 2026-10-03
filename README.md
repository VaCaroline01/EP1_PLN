# EP1 - Processamento de Linguagem Natural

Projeto desenvolvido para o EP1 da disciplina de Processamento de Linguagem Natural (PLN), com o objetivo de construir um modelo capaz de classificar automaticamente textos em três classes de clareza: `c1`, `c234` e `c5`.

O projeto compara uma abordagem baseline utilizando **TF-IDF + Regressão Logística** com uma abordagem baseada no modelo de linguagem **BERTimbau**.

---

## Objetivo

O objetivo do trabalho é desenvolver um modelo de classificação de textos capaz de superar o desempenho do baseline e, posteriormente, utilizar o melhor modelo para classificar os textos presentes no conjunto de teste.

Foram utilizadas as seguintes classes:

- `c1`
- `c234`
- `c5`

---

## Conjunto de dados

O conjunto de treinamento (`train.xlsx`) possui **20.092 registros** e duas colunas:

- `resp_text`: texto utilizado como entrada do modelo;
- `clarity`: classe associada ao texto.

Distribuição das classes no conjunto de treinamento:

| Classe | Quantidade |
|---|---:|
| c5 | 6.892 |
| c234 | 6.853 |
| c1 | 6.347 |
| **Total** | **20.092** |

Para o desenvolvimento e comparação dos modelos, o conjunto `train.xlsx` foi inicialmente dividido em:

- **80% para treinamento:** 16.073 registros;
- **20% para validação:** 4.019 registros.

O conjunto de teste (`test1.xlsx`) possui **900 registros** e não foi utilizado durante o treinamento ou escolha do modelo.

---

## 1. Baseline - TF-IDF + Regressão Logística

O primeiro modelo utilizado foi o baseline composto por:

**TF-IDF + Regressão Logística**

O TF-IDF transforma os textos em representações numéricas de acordo com a importância das palavras no conjunto de documentos. Essas representações são utilizadas como entrada para o classificador de Regressão Logística.

### Resultado do baseline

| Métrica | Resultado |
|---|---:|
| Acurácia | **44,99%** |
| F1 Macro | **0,4474** |

Esse resultado foi utilizado como referência para avaliar o modelo proposto.

---

## 2. BERTimbau V1

Como abordagem mais avançada, foi utilizado o **BERTimbau**, modelo BERT pré-treinado para a língua portuguesa.

Modelo utilizado:

`neuralmind/bert-base-portuguese-cased`

Na primeira configuração, foram utilizados:

- `MAX_LENGTH = 128`;
- `BATCH_SIZE = 16`;
- learning rate de `2e-5`;
- 3 épocas de treinamento.

### Resultados

| Época | Acurácia | F1 Macro |
|---|---:|---:|
| 1 | 44,61% | 0,4144 |
| 2 | **44,91%** | 0,4420 |
| 3 | 44,51% | **0,4458** |

A primeira versão do BERTimbau não conseguiu superar a acurácia do baseline.

---

## 3. Análise do tamanho dos textos

Após o primeiro experimento, foi analisada a quantidade de tokens dos textos do conjunto de treinamento.

Os resultados foram:

| Estatística | Tokens |
|---|---:|
| Média | 253 |
| Mediana | 182 |
| Percentil 75% | 331 |
| Percentil 90% | 535 |
| Percentil 95% | 718 |
| Máximo | 4.429 |

Também foi identificado que:

- **63,2%** dos textos possuem mais de 128 tokens;
- **35,8%** possuem mais de 256 tokens;
- **10,8%** possuem mais de 512 tokens.

Dessa forma, a configuração inicial com `MAX_LENGTH = 128` truncava uma parcela significativa dos textos.

A partir dessa análise foi desenvolvida uma segunda configuração do BERTimbau.

---

## 4. BERTimbau V2

A segunda versão aumentou o tamanho máximo das sequências para **256 tokens** e adicionou técnicas de otimização e regularização ao treinamento.

Configuração utilizada:

| Parâmetro | Valor |
|---|---|
| Modelo | BERTimbau Base |
| MAX_LENGTH | 256 |
| BATCH_SIZE | 8 |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Warmup | 10% |
| Épocas | 3 |

Também foram utilizados:

- AdamW;
- scheduler linear;
- gradient clipping;
- armazenamento do melhor checkpoint durante a validação.

### Resultados

| Época | Acurácia | F1 Macro |
|---|---:|---:|
| 1 | 44,79% | 0,4206 |
| 2 | 45,91% | 0,4489 |
| 3 | **45,93%** | **0,4561** |

A terceira época apresentou o melhor resultado.

---

## Comparação dos modelos

| Modelo | Acurácia | F1 Macro |
|---|---:|---:|
| TF-IDF + Regressão Logística | 44,99% | 0,4474 |
| BERTimbau V1 | 44,91% | 0,4420 |
| **BERTimbau V2** | **45,93%** | **0,4561** |

O BERTimbau V2 apresentou uma melhora de aproximadamente **0,94 ponto percentual de acurácia** em relação ao baseline.

Por apresentar a maior acurácia entre as configurações avaliadas, o **BERTimbau V2 foi selecionado como modelo final**.

---

## 5. Treinamento final

Após a escolha da melhor configuração, foi realizado um novo treinamento do BERTimbau utilizando **100% dos 20.092 registros do `train.xlsx`**.

Foram mantidos os hiperparâmetros definidos no BERTimbau V2 e realizadas três épocas de treinamento.

Loss médio observado durante o treinamento final:

| Época | Loss médio |
|---|---:|
| 1 | 1,0688 |
| 2 | 0,9977 |
| 3 | 0,8854 |

A redução progressiva do loss indica que o modelo reduziu o erro sobre os dados de treinamento ao longo das épocas.

A acurácia de **45,93%** apresentada anteriormente corresponde à etapa de validação 80/20. Como o treinamento final utiliza 100% dos dados rotulados, não há um conjunto de validação separado nessa etapa.

---

## 6. Classificação do conjunto de teste

Após o treinamento final, o modelo foi utilizado para classificar os **900 textos** presentes no arquivo `test1.xlsx`.

O conjunto de teste foi utilizado exclusivamente para inferência, não participando do treinamento nem da escolha dos hiperparâmetros.

Distribuição das previsões:

| Classe | Quantidade |
|---|---:|
| c234 | 349 |
| c5 | 303 |
| c1 | 248 |
| **Total** | **900** |

O arquivo contendo as classificações finais está disponível em:

`dados/test1_classificado.xlsx`

---

## Estrutura do repositório

```text
EP1_PLN/
│
├── dados/
│   └── test1_classificado.xlsx
│
├── notebooks/
│   └── ep_1.ipynb
│
├── README.md
└── requirements.txt
```

### `notebooks/ep_1.ipynb`

Notebook contendo o desenvolvimento do projeto, incluindo:

- carregamento e análise dos dados;
- divisão entre treinamento e validação;
- baseline TF-IDF + Regressão Logística;
- treinamento do BERTimbau;
- análise do tamanho dos textos;
- otimização do BERTimbau;
- comparação dos modelos;
- treinamento final;
- classificação do conjunto de teste.

### `dados/test1_classificado.xlsx`

Arquivo contendo os 900 textos do conjunto de teste com as classes previstas pelo modelo final.

### `requirements.txt`

Contém as principais bibliotecas necessárias para executar o projeto.

---

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Transformers
- BERTimbau
- Google Colab

O treinamento do BERTimbau foi realizado utilizando GPU **NVIDIA Tesla T4** disponibilizada pelo Google Colab.

---

## Como executar

### 1. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 2. Abrir o notebook

O desenvolvimento completo está disponível em:

```text
notebooks/ep_1.ipynb
```

O notebook pode ser executado utilizando **Google Colab** ou outro ambiente Python compatível com as dependências do projeto.

Para o treinamento do BERTimbau, recomenda-se a utilização de GPU.

---

## Resultado final

O experimento apresentou os seguintes resultados principais:

**Baseline — TF-IDF + Regressão Logística**

Acurácia: **44,99%**

**Modelo proposto — BERTimbau V2**

Acurácia: **45,93%**

F1 Macro: **0,4561**

O BERTimbau V2 apresentou o melhor desempenho entre as configurações avaliadas e foi utilizado para gerar as previsões finais do conjunto de teste.
