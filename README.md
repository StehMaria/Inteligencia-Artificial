# Algoritmos e Paradigmas de Machine Learning 🧠

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Fundamentos-blue?style=for-the-badge)
![Data Science](https://img.shields.io/badge/Data%20Science-Pré--Processamento-success?style=for-the-badge)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-IAs%20Generativas-orange?style=for-the-badge)

Este repositório serve como um guia abrangente e estruturado sobre o ciclo de vida do Aprendizado de Máquina. Ele cobre desde a preparação dos dados até os motores matemáticos e métricas de avaliação dos principais paradigmas de Inteligência Artificial.

---

## 📌 Índice
1. [Pré-Processamento e Dados](#1-pré-processamento-e-dados)
2. [Redução de Dimensionalidade (PCA)](#2-redução-de-dimensionalidade-pca)
3. [Aprendizado Supervisionado](#3-aprendizado-supervisionado)
4. [Aprendizado Não Supervisionado](#4-aprendizado-não-supervisionado)
5. [Aprendizado Semi-Supervisionado](#5-aprendizado-semi-supervisionado)
6. [Aprendizado Ativo (Active Learning)](#6-aprendizado-ativo-active-learning)
7. [Aprendizado por Reforço](#7-aprendizado-por-reforço-reinforcement-learning)
8. [Deep Learning e IAs Generativas](#8-deep-learning-e-ias-generativas)

---

## 🗄️ 1. Pré-Processamento e Dados

A fase de preparação dos dados é o verdadeiro "trabalho braçal" da Ciência de Dados (*Garbage in, Garbage out*). Antes de processar, precisamos entender a natureza de cada coluna do nosso banco de dados.

### Categorização dos Atributos
* **Qualitativos (Categóricos):** Descrevem qualidades (não matemáticos).
  * *Nominais:* Sem hierarquia (ex: Cor dos olhos, Estado Civil).
  * *Ordinais:* Possuem ordem ou ranking (ex: P, M, G, GG; Patente militar).
* **Quantitativos (Numéricos):** Representam quantidades mensuráveis.
  * *Discretos:* Valores inteiros / contagens (ex: Número de filhos).
  * *Contínuos:* Valores reais / medições (ex: Altura 1.75m, Salário).

### Técnicas de Processamento
Algoritmos baseados em distância (KNN) ou gradiente (Redes Neurais) são extremamente sensíveis à escala dos dados.
* **Limpeza (Data Cleansing):** Tratar dados faltantes e remover *outliers*.
* **Normalização (Min-Max):** Espreme todos os valores para o intervalo exato entre **0 e 1**.
* **Padronização (Z-Score):** Ajusta os dados para **Média 0** e **Desvio Padrão 1**. Resistente a *outliers*.
* **Binarização (One-Hot Encoding):** Transforma variáveis qualitativas em colunas binárias (0 ou 1).

### Medidas de Distância / Similaridade
* **Euclidiana:** Distância em linha reta geométrica (Teorema de Pitágoras).
* **Manhattan:** Distância em "quarteirões" (soma das diferenças absolutas).
* **Cosseno:** Mede o ângulo entre dois vetores (excelente para textos).

---

## 📉 2. Redução de Dimensionalidade (PCA)

O **PCA (Principal Component Analysis)** condensa dezenas de colunas em poucas novas colunas (Componentes Principais), preservando o máximo de informação (variância) possível, evitando a "Maldição da Dimensionalidade".

* **Matriz de Covariância:** Verifica quais atributos andam juntos.
* **Autovetores (Eigenvectors):** Apontam a direção no espaço onde os dados estão mais espalhados.
* **Autovalores (Eigenvalues):** O grau de importância de cada Autovetor (ex: "O Componente 1 carrega 80% da informação").

---

## 🎯 3. Aprendizado Supervisionado

O algoritmo aprende com um "gabarito" (dados rotulados). O objetivo é encontrar uma função matemática que mapeie a entrada para a saída. O motor faz o trabalho do início ao fim.

### A. Classificação (Prever Categorias)
* **Regressão Logística:** Usa a **Função Sigmoide** para retornar a probabilidade de uma classe (0% a 100%).
* **Árvores de Decisão:** Usam **Entropia de Shannon** e **Impureza de Gini** para calcular os melhores cortes.
* **SVM (Support Vector Machines):** Usa **Multiplicadores de Lagrange** e o **Truque do Kernel** para traçar a margem máxima entre classes.
* **Naive Bayes:** Baseado em **Probabilidade Condicional** (Teorema de Bayes). Assume independência entre características.
* **KNN:** Memoriza os dados e classifica baseado na distância dos *K* vizinhos mais próximos.
* **Ensembles (Random Forest, XGBoost):** Combinação de múltiplas árvores fracas votando juntas.  

**Avaliação:** Matriz de Confusão, Acurácia, Precisão, Recall, F1-Score.

### B. Regressão (Prever Números Contínuos)
* **Algoritmos:** Regressão Linear (OLS / Gradiente Descendente), Regressões Regularizadas (Ridge, Lasso), SVR e Decision Tree Regressor.  

**Avaliação:** MSE (Erro Quadrático Médio), MAE (Erro Absoluto Médio), R² (Coeficiente de Determinação).

---

## 🧩 4. Aprendizado Não Supervisionado

Não há gabarito. O algoritmo é jogado em um mar de dados puros e deve encontrar padrões, estruturas ou anomalias baseando-se em estatística espacial.

### A. Agrupamento (Clustering)
* **K-Means:** Calcula o **Centroide** (média aritmética) para minimizar o SSE intra-cluster. Assume grupos esféricos.
* **Hierárquico:** Usa **Critérios de Linkage** para fundir dados passo a passo (Dendrograma).
* **DBSCAN:** Baseado na densidade. Conta pontos dentro de um raio (Epsilon). Lida bem com ruídos e formatos arbitrários.  

**Avaliação Interna:** Índice de Silhueta, Coesão (WSS) e Separação (BSS).

### B. Detecção de Anomalias
* **Isolation Forest:** Usa árvores de decisão cortando dados aleatoriamente.
* **One-Class SVM:** Usa o motor SVM fechado em uma fronteira circular.
* **Local Outlier Factor (LOF):** Usa a matemática do KNN.

### C. Regras de Associação (Eventos conjuntos)
* **Apriori / FP-Growth:** Focado em itens de cesta (ex: "Se compra X, compra Y").
* **Matemática:** Suporte (Frequência), Confiança (Probabilidade Condicional) e Lift (Razão de dependência).

---

## 🌗 5. Aprendizado Semi-Supervisionado

Estratégias para preencher as lacunas quando há muitos dados sem rótulo e poucos dados com rótulo.

* **Self-Training:** O modelo treina com dados rotulados, prevê os não rotulados e aceita as maiores confianças como verdadeiras para o próximo ciclo.
  * *Motores:* Regressão Logística, Random Forest, Redes Neurais, Naive Bayes.
* **Co-Training:** Dois modelos diferentes (ex: Naive Bayes + Árvore de Decisão) olham para características diferentes e ensinam um ao outro.
* **Label Propagation:** Algoritmo baseado em grafos. A cor do rótulo "vaza" para os vizinhos mais próximos proporcionalmente à distância (Kernel RBF).
* **TSVM (Transductive SVM):** Modificação matemática do SVM forçando a reta a passar pelas regiões mais vazias entre os dados não rotulados.

---

## 🙋 6. Aprendizado Ativo (Active Learning)

O foco é interagir com um Oráculo (humano). O motor precisa saber dizer "eu não tenho certeza" e pedir ajuda apenas para os dados mais difíceis.

* **Uncertainty Sampling (Amostragem por Incerteza):** O modelo pede ajuda no dado onde tem maior confusão (Entropia máxima ou menor margem de confiança).
  * *Motores:* Probabilísticos (Regressão Logística, Redes Neurais).
* **Query-by-Committee:** Um comitê de algoritmos (ex: 1 Árvore, 1 SVM, 1 KNN). Se houver grande discordância entre eles, o oráculo é acionado.
* **Expected Model Change:** Pede o rótulo do dado que causaria a maior mudança matemática nos pesos do modelo (baseado em gradientes de Redes Neurais).

---

## 🎮 7. Aprendizado por Reforço (Reinforcement Learning)

Baseado em tentativa e erro, lidando com o binômio Ação x Recompensa no tempo.

* **Q-Learning Clássico / SARSA:** Constrói tabelas (Q-Table) mapeando a recompensa esperada. Usa a **Equação de Bellman** (Taxa de Aprendizado $\alpha$ e Fator de Desconto $\gamma$).
  * *Motores:* Matrizes e Tabelas de Hash (Programação Dinâmica).
* **Deep Q-Network (DQN) / PPO / REINFORCE:** Unem Q-Learning a Redes Neurais para prever valores de tabelas infinitas ou ajustar políticas diretamente.
  * *Motores:* Redes Neurais (MLPs, CNNs).
* **Avaliação:** Recompensa Cumulativa Média por episódio.

---

## 🎨 8. Deep Learning e IAs Generativas (Auto-Supervisionado)

Território absoluto das Redes Neurais Artificiais massivas. Trabalham com soma ponderada, viés, funções de ativação e **Backpropagation** (Regra da Cadeia).

* **Geração de Texto (LLMs):**
  * *Motores:* **Transformers** (Mecanismo de Auto-Atenção via matrizes Query, Key, Value).
* **Geração de Imagens:**
  * *Motores:* **Modelos de Difusão** (adicionam e removem ruído via U-Net) e **GANs** (Redes Adversárias - Gerador vs. Discriminador).
* **Avaliação:** Loss Functions (Cross-Entropy Loss, MSE).



1. Aprendizado Semi-Supervisionado
As estratégias aqui precisam de motores que consigam lidar com a mistura de dados com e sem rótulo.

Self-Training (Auto-treinamento):

Motores aceitos: Basicamente qualquer classificador que retorne um nível de probabilidade ou confiança, não apenas a classe final.

Principais escolhas: Naive Bayes, Regressão Logística, Random Forest (Árvores) e Redes Neurais. (O KNN é pouco usado aqui porque não tem uma "confiança" matemática tão estável quanto probabilidades puras).

Co-Training (Co-treinamento):

Motores aceitos: Requer o uso de dois modelos simultâneos, preferencialmente de naturezas matemáticas diferentes, para que um não cometa o mesmo erro que o outro.

Principais escolhas: A combinação mais clássica é usar um Naive Bayes (olhando para o texto de uma página web) trabalhando junto com uma Árvore de Decisão (olhando para os links da página). SVM e Regressão Logística também são combinados frequentemente.

Label Propagation / Label Spreading (Propagação de Rótulos):

Motores aceitos: Esta estratégia não usa os modelos de classificação clássicos. Ela usa Motores de Grafos e Matrizes de Distância.

Principais escolhas: A matemática por baixo é idêntica à do KNN (K-Vizinhos Mais Próximos) combinada com funções de raio vetorial (Kernel RBF). O cálculo é puramente baseado em distância física geométrica entre os pontos.

TSVM (Transductive SVM):

Motores aceitos: Exclusivamente o SVM (Support Vector Machine). É uma modificação matemática direta na fórmula do hiperplano do SVM para forçá-lo a desviar de aglomerações de dados não rotulados.

2. Aprendizado Ativo (Active Learning)
A estratégia aqui é medir a "dúvida" do modelo. O motor precisa saber dizer "eu não tenho certeza".

Uncertainty Sampling (Amostragem por Incerteza):

Motores aceitos: Modelos probabilísticos.

Principais escolhas: Regressão Logística, Random Forest, Redes Neurais e SVM Probabilístico.

Query-by-Committee (Consulta por Comitê):

Motores aceitos: Um conjunto (ensemble) de algoritmos distintos (para que eles possam discordar entre si).

Principais escolhas: Junta-se, por exemplo, 1 Árvore de Decisão + 1 SVM + 1 Regressão Logística. Se a Árvore diz "A", mas o SVM e a Regressão dizem "B", o algoritmo pede ajuda ao humano.

Expected Model Change (Mudança Esperada):

Motores aceitos: Modelos baseados em pesos e gradientes (cálculo diferencial).

Principais escolhas: Redes Neurais (onde se calcula o tamanho do ajuste nos neurônios) e Regressão Logística. Árvores de Decisão não servem para isso.

3. Aprendizado Supervisionado
Como é o modelo mais direto (aprende com o gabarito), a estratégia e o motor são praticamente a mesma coisa. O motor faz o trabalho do início ao fim.

Classificação (Separar categorias): Regressão Logística, Árvores de Decisão, Random Forest, XGBoost, SVM, KNN, Naive Bayes e Redes Neurais.

Regressão (Prever números contínuos): Regressão Linear, SVR (Support Vector Regression), Decision Tree Regressor e Redes Neurais.

4. Aprendizado Não Supervisionado
Os motores são construídos puramente em cima de estatística espacial e distâncias, já que não possuem gabarito para correção.

Clustering (Agrupamento):

K-Means: Usa matemática de centro de massa vetorial.

DBSCAN: Usa a matemática do KNN (raio de distância e contagem de vizinhos) para achar zonas densas.

Detecção de Anomalias:

Isolation Forest: Usa exclusivamente o motor da Árvore de Decisão (cortando os dados aleatoriamente).

One-Class SVM: Usa exclusivamente o motor do SVM fechado em uma fronteira circular.

Local Outlier Factor (LOF): Usa a matemática do KNN.

Redução de Dimensionalidade:

PCA: Usa Álgebra Linear pura (Autovetores e Matrizes de Covariância).

Autoencoders: Usa exclusivamente Redes Neurais Artificiais.

5. Aprendizado por Reforço
Os motores aqui lidam com estados, ações e recompensas. A evolução da área dividiu os motores entre tabelas simples e arquiteturas profundas.

Q-Learning Clássico / SARSA:

Motores aceitos: Matrizes e Tabelas de Hash (Programação Dinâmica). Não usa algoritmos preditivos clássicos; usa memória tabular estática.

Deep Q-Network (DQN) / PPO / Actor-Critic:

Motores aceitos: Exclusivamente Redes Neurais (Deep Learning). Usa MLPs (Perceptrons de Múltiplas Camadas) para tomar decisões lógicas, ou CNNs (Redes Convolucionais) se o robô/agente precisar "enxergar" o ambiente (como jogar Mario vendo a tela).

6. IAs Generativas (Auto-Supervisionado)
Este é o território exclusivo e absoluto das Redes Neurais Artificiais (Deep Learning). Nenhum outro motor matemático tradicional (Árvore, SVM, KNN) consegue gerar textos ou imagens.

Geração de Texto (LLMs como ChatGPT):

Motor exclusivo: Redes Neurais com arquitetura Transformer (baseadas no mecanismo de auto-atenção matemática).

Geração de Imagens Fotorrealistas:

Motores: Modelos de Difusão (usa Redes Neurais Convolucionais conhecidas como U-Net para remover ruído) ou GANs (usa duas Redes Neurais brigando entre si).




Aqui estão os principais algoritmos (e estratégias) que dão vida a cada um desses paradigmas.

É importante notar que, em algumas dessas áreas (como no Aprendizado Ativo), os "algoritmos" são, na verdade, **heurísticas de busca** que rodam "por cima" de algoritmos tradicionais (como Árvores de Decisão ou SVMs).

---

### 1. Aprendizado Semi-Supervisionado

Os algoritmos aqui tentam preencher as lacunas dos dados não rotulados usando a pouca informação rotulada que possuem.

* **Self-Training (Auto-treinamento):** O algoritmo mais clássico. Um modelo base (como Naive Bayes ou KNN) é treinado com os poucos dados rotulados. Em seguida, ele prevê os rótulos da base sem rótulo. As previsões nas quais o modelo tem maior confiança são aceitas como "verdadeiras" e adicionadas à base de treino para o próximo ciclo.
* **Co-Training (Co-treinamento):** Usa dois modelos diferentes olhando para características (features) diferentes dos mesmos dados. O que o Modelo A aprende com alta confiança, ele ensina para o Modelo B, e vice-versa.
* **Label Propagation (Propagação de Rótulos):** Um algoritmo baseado em grafos. Imagine que cada dado é um nó em uma rede. Os dados rotulados "espalham" a cor do seu rótulo para os vizinhos mais próximos e parecidos (dados não rotulados) até que toda a rede esteja colorida.
* **TSVM (Transductive Support Vector Machines):** Uma adaptação do tradicional SVM. Em vez de apenas traçar uma reta que separe as classes rotuladas, ele traça uma reta que também tenta cortar as regiões mais vazias (de menor densidade) entre os dados não rotulados.

---

### 2. Aprendizado Ativo (Active Learning)

Como o foco aqui é "o que perguntar ao oráculo (humano)", os algoritmos são essencialmente **Estratégias de Consulta (Query Strategies)**. Eles usam um modelo base (como Regressão Logística ou Random Forest) e aplicam regras matemáticas para encontrar a maior dúvida.

* **Uncertainty Sampling (Amostragem por Incerteza):** O modelo escolhe os dados sobre os quais está mais confuso. Pode ser calculado de três formas principais:
* *Least Confidence (Menor Confiança):* Pega o dado cuja predição da classe vencedora foi a mais baixa.
* *Margin Sampling (Margem):* Pega o dado onde a diferença de probabilidade entre a 1ª e a 2ª classe mais provável é mínima (ou seja, o modelo está "em cima do muro" entre duas classes).
* *Entropy (Entropia):* Usa a fórmula de Shannon para achar a amostra com a maior confusão geral entre todas as classes.


* **Query-by-Committee (Consulta por Comitê):** Cria-se um comitê de vários modelos diferentes (ex: uma Árvore, um SVM e um KNN). O algoritmo manda para o humano rotular o dado que causou a maior briga (maior discordância) entre os membros do comitê.
* **Expected Model Change (Mudança Esperada do Modelo):** O algoritmo calcula matematicamente qual dado, se recebesse um rótulo agora, causaria a maior mudança nos pesos da rede neural ou na estrutura do modelo atual.

---

### 3. Aprendizado por Reforço (Reinforcement Learning)

Aqui temos uma família rica de algoritmos divididos pela forma como o agente calcula a sua próxima jogada no ambiente.

* **Q-Learning:** O algoritmo clássico baseado em Valor (Model-Free). Ele constrói uma grande tabela (Q-Table) que mapeia qual é a recompensa esperada para cada ação em cada estado possível.
* **SARSA (State-Action-Reward-State-Action):** Muito parecido com o Q-Learning, mas é mais "cauteloso". Ele atualiza seus valores baseando-se na ação que ele *realmente* executou, enquanto o Q-Learning sempre assume que fará a melhor ação possível no futuro.
* **DQN (Deep Q-Network):** A revolução que uniu Redes Neurais com Q-Learning. Em vez de usar uma tabela (o que é impossível em jogos complexos, pois as opções são infinitas), usa uma rede neural para adivinhar os valores da tabela. (Foi o algoritmo que a DeepMind usou para zerar jogos de Atari sozinhos).
* **REINFORCE:** Um algoritmo baseado puramente em Política. Ele não tenta adivinhar o valor do estado; ele simplesmente ajusta a probabilidade de tomar uma ação. Se a ação gerou vitória, ele aumenta a probabilidade de repeti-la.
* **PPO (Proximal Policy Optimization):** É o estado da arte atual (usado para treinar os robôs da Boston Dynamics e as IAs da OpenAI). É um modelo "Ator-Crítico" (híbrido). Ele atualiza o comportamento do agente de forma suave, impedindo que a IA mude de ideia tão drasticamente a ponto de "desaprender" o que já sabia.

---

### 4. IAs Generativas (Modelos Auto-Supervisionados Profundos)

Embora não sejam um paradigma de aprendizado clássico da taxonomia antiga, do ponto de vista de algoritmos e arquiteturas, a divisão atual do mercado é fortemente baseada no tipo de mídia gerada:

* **Transformers (Arquitetura central para Texto/LLMs):** Criados pelo Google em 2017. O grande segredo é o "Mecanismo de Atenção", que permite ao algoritmo entender o contexto de uma palavra olhando para todas as outras palavras da frase ao mesmo tempo (É o motor por trás do GPT, BERT, Llama).
* **GANs (Redes Adversárias Generativas):** Usadas para gerar rostos fotorrealistas e *deepfakes*. Consistem em duas redes neurais brigando: O *Gerador* tenta criar uma imagem falsa perfeita, e o *Discriminador* tenta descobrir se a imagem é real ou falsa. O treino acaba quando o gerador engana o discriminador.
* **Modelos de Difusão (Diffusion Models):** Substituíram as GANs na geração moderna de arte (DALL-E 3, Midjourney, Stable Diffusion). O algoritmo pega uma imagem real, a destrói gradualmente adicionando "chuvisco" (ruído estático) até virar um borrão completo, e depois treina uma rede neural para fazer o caminho inverso (remover o ruído até formar uma imagem nova).
* **VAEs (Variational Autoencoders):** Algoritmos mais antigos e matematicamente rígidos, usados hoje para tarefas como geração de moléculas na biotecnologia ou compressão avançada de dados. Eles comprimem o dado em uma representação latente (menor) e tentam reconstruí-lo gerando variações.



Para completar o quadro taxonômico da Inteligência Artificial (especialmente alinhado com a visão clássica do livro da Faceli e da literatura canônica), vamos detalhar os dois pilares mais tradicionais do Machine Learning: o **Supervisionado** e o **Não Supervisionado**.

Aqui, a divisão primária ocorre pelas **tarefas** que eles resolvem.

---

### 1. Aprendizado Supervisionado (Supervised Learning)

O algoritmo aprende com um "gabarito" (dados rotulados). O objetivo é encontrar uma função matemática que mapeie a entrada (features) para a saída (rótulo/alvo) para poder prever o futuro. Ele se divide em duas grandes tarefas: **Classificação** (prever categorias) e **Regressão** (prever números contínuos).

#### Algoritmos de Classificação (Prever "O Quê" ou "Qual Categoria")

* **Regressão Logística (Logistic Regression):** Apesar do nome ter "regressão", é o algoritmo base para classificação binária. Ele usa uma curva em formato de "S" (função sigmoide) para espremer os dados entre 0 e 1, retornando a probabilidade de um dado pertencer a uma classe (ex: 80% de chance de ser spam).
* **K-Nearest Neighbors (KNN - K-Vizinhos Mais Próximos):** Um algoritmo "preguiçoso". Ele não treina um modelo matemático complexo; ele simplesmente memoriza os dados. Quando chega um dado novo, ele olha para os "K" vizinhos mais próximos no gráfico e faz uma votação. Se a maioria ao redor for da classe A, ele será classificado como A.
* **Naive Bayes:** Baseado no Teorema de Bayes da probabilidade estatística. Ele é "ingênuo" (naive) porque assume que todas as características do dado são independentes entre si. É extremamente rápido e, historicamente, o padrão-ouro inicial para filtros de spam e análise de sentimentos em textos.
* **Support Vector Machines (SVM - Máquinas de Vetores de Suporte):** Tenta desenhar uma "rua" (hiperplano) entre as classes no gráfico. O objetivo do SVM é fazer com que essa rua seja a mais larga possível (margem máxima). Se os dados não forem separáveis por uma linha reta, ele usa o "Truque do Kernel" para dobrar o espaço 3D e cortá-los.
* **Árvores de Decisão (Decision Trees):** Aprende dividindo os dados através de perguntas de "Sim/Não" baseadas nas características (ex: "Idade > 30?", "Renda > 5000?").
* **Métodos Ensemble (Random Forest, XGBoost, LightGBM):** Não são algoritmos novos, mas sim a combinação de centenas ou milhares de Árvores de Decisão fracas votando juntas. É a técnica dominante em competições de dados estruturados (Kaggle).

#### Algoritmos de Regressão (Prever "Quanto")

* **Regressão Linear (Linear Regression):** Tenta traçar a "linha de melhor ajuste" que passe o mais perto possível de todos os pontos de dados (ex: prever o preço de uma casa baseado na sua metragem).
* **Regressões Regularizadas (Ridge, Lasso, Elastic Net):** São evoluções da Regressão Linear que penalizam o modelo matematicamente para evitar o *overfitting* (quando o modelo decora os dados de treino em vez de aprender o padrão geral).
* **Support Vector Regression (SVR):** É o SVM adaptado para números contínuos. Em vez de tentar traçar uma reta para separar os dados, ele tenta traçar um "tubo" que englobe o máximo de pontos possíveis, ignorando os erros pequenos.

---

### 2. Aprendizado Não Supervisionado (Unsupervised Learning)

Aqui não há gabarito. O algoritmo é jogado em um mar de dados puros e sua missão é encontrar padrões ocultos, estruturas ou anomalias sozinho. Divide-se em três tarefas principais: **Agrupamento (Clustering)**, **Redução de Dimensionalidade** e **Regras de Associação**.

#### Algoritmos de Agrupamento (Clustering)

O objetivo é separar os dados em "panelinhas" onde itens do mesmo grupo sejam muito parecidos, e itens de grupos diferentes sejam muito distintos.

* **K-Means:** O algoritmo mais famoso. Você diz para ele quantos grupos quer (o "K"). Ele joga "K" centros aleatórios no gráfico e vai ajustando a posição deles até encontrar o centro de massa de cada grupo. Ele assume que os grupos são redondos (esféricos).
* **DBSCAN:** Agrupamento baseado em densidade. Diferente do K-Means, você não precisa dizer quantos grupos quer. Ele sai procurando áreas com muitos pontos concentrados. A grande vantagem é que ele acha grupos de qualquer formato (uma meia-lua, por exemplo) e isola automaticamente dados esquisitos como ruído/outliers.
* **Agrupamento Hierárquico (Hierarchical Clustering):** Começa tratando cada dado como um grupo isolado e vai fundindo os mais próximos passo a passo, construindo uma grande árvore (um Dendrograma). É ótimo para taxonomia (como agrupar espécies de animais pela genética).

#### Algoritmos de Redução de Dimensionalidade

Se você tem uma planilha com 10.000 colunas (como dados de DNA ou pixels de uma imagem), o algoritmo reduz isso para 2 ou 3 colunas mantendo a essência da informação, permitindo a visualização em gráficos normais e acelerando o processamento.

* **PCA (Principal Component Analysis):** O método linear clássico. Ele roda eixos através dos dados tentando encontrar o ângulo que preserva a maior variação (diferença) entre eles.
* **t-SNE (t-Distributed Stochastic Neighbor Embedding):** Método não linear fantástico para criar visualizações 2D ou 3D de dados complexos. Ele garante que pontos que estavam perto no espaço de 10.000 dimensões continuem perto na tela de 2 dimensões.
* **UMAP:** A evolução moderna do t-SNE. Faz a mesma coisa que o t-SNE, mas de forma muito mais rápida matematicamente e preservando melhor a estrutura global dos dados.

#### Algoritmos de Regras de Associação

Focado em descobrir quais eventos acontecem juntos.

* **Apriori / FP-Growth:** São os clássicos "algoritmos de cesta de supermercado". Eles varrem milhões de notas fiscais (ou carrinhos de e-commerce) para extrair regras como: *"Se um cliente compra Fraldas e Cerveja, há 80% de chance de ele também comprar Salgadinho"*. O sistema de recomendação inicial da Amazon foi fortemente baseado nessas regras estruturais.


1. O Pilar dos Dados e Pré-Processamento (Aulas 02, 03 e Listas 1)
Antes de jogar os dados nos algoritmos, você precisa entender o que eles são e prepará-los. Os algoritmos baseados em distância (como KNN e K-Means) dependem 100% desta etapa.

Tipologia dos Atributos (Listas 1_1 e 1_2):

Qualitativos: Nominais (ex: cor dos olhos, opaco/translúcido) e Ordinais (ex: patente militar, medalhas de ouro/prata/bronze).

Quantitativos: Discretos (ex: número de pacientes) e Contínuos (ex: altura, peso, ângulos).

Medidas de Similaridade (O coração do KNN e Clustering):

Como os algoritmos sabem o que é parecido? Usando métricas como Distância Euclidiana, Distância de Manhattan, Similaridade de Cosseno (ótimo para textos) e Correlação.

Transformação e Pré-processamento:

Normalização/Padronização (MinMax): Você não pode colocar no mesmo algoritmo uma coluna de "Idade" (0 a 100) e uma coluna de "Salário" (1.000 a 50.000), senão o salário vai esmagar a idade matematicamente. Padronizar é colocar tudo na mesma escala (ex: média 0, desvio padrão 1).

Agregação e Seleção: Reduzir o tamanho dos dados agrupando objetos ou removendo atributos inúteis.

2. O Aprofundamento no Não-Supervisionado (Aulas 10 a 13 e SADM_2010)
Seus materiais dão um foco gigantesco em Clustering (Agrupamento). Eles dividem esse mundo de forma mais rigorosa do que a taxonomia básica:

Métodos Particionais (ex: K-Means): Constroem partições (grupos rígidos e exclusivos). Você precisa dizer o número de grupos (K) de antemão. São rápidos, mas assumem que os grupos são esféricos (redondos).

Métodos Hierárquicos (Aglomerativos / Bottom-Up): Constroem uma hierarquia (Dendrograma). Cada ponto começa como seu próprio grupo e eles vão se fundindo (merge) com os mais próximos até virarem um grupo só (usando técnicas de linkage como Single, Complete ou Average). Vantagem: Você não precisa chutar o número de "K" no início.

Métodos Baseados em Densidade (ex: DBSCAN): Como o material de Hartigan mostra, o K-Means falha miseravelmente se os grupos tiverem formatos esquisitos (como duas meias-luas). Algoritmos de densidade acham formatos arbitrários e ignoram "ruídos" (pontos anômalos isolados).

Validação de Clusters: Como não temos gabarito (já que é não-supervisionado), como saber se o K-Means agrupou certo? Usamos índices matemáticos internos (como a Silhueta ou Coesão/Separação) para ver se os grupos estão bem juntinhos por dentro e bem afastados por fora.

3. O Pilar da Avaliação e Confiança (Aula 04, Bishop e Mitchell)
O livro do Mitchell e o do Bishop batem muito forte nesta tecla, assim como a Aula 04 de Classificação. Não basta o algoritmo aprender, ele precisa provar que generaliza para o mundo real.

O Problema do Overfitting: Quando o modelo "decora" os dados de treino, mas erra tudo nos dados novos.

Curvas de Aprendizado (Learning Curves): Gráficos que mostram como a acurácia do modelo varia conforme aumentamos o tamanho da amostra (mostrando o impacto de small sample sizes).

Estimativas de Confiabilidade: Em vez de olhar só o erro no treino, os dados devem ser divididos em Treino (para o modelo aprender), Validação (para ajustar hiperparâmetros) e Teste (para a prova final). O material do C4.5 (Witten) menciona até estimativas pessimistas baseadas em distribuições binomiais para garantir a confiança das regras geradas.


Para organizar essa "sopa de letrinhas" matemática, vamos dividir o mundo do Machine Learning em duas grandes categorias de cálculos para cada paradigma:

1. **Os Cálculos do Motor (Treinamento/Otimização):** A matemática que o algoritmo usa "por baixo do capô" para aprender.
2. **Os Cálculos de Avaliação (Teste/Validação):** As métricas que nós usamos para medir se o modelo ficou bom ou ruim.

Aqui está o guia completo e organizado de todos os cálculos envolvidos:

---

### 0. Pré-Processamento (A Base de Tudo)

Antes de qualquer paradigma, os dados precisam ser traduzidos para a matemática.

* **Cálculos de Distância/Similaridade (Essencial para KNN, K-Means e DBSCAN):**
* *Euclidiana:* A distância em linha reta geométrica (Teorema de Pitágoras).
* *Manhattan:* A distância andando em "quarteirões" (soma das diferenças absolutas).
* *Cosseno:* Mede o ângulo entre dois vetores (usado para textos e documentos).


* **Transformações de Escala:**
* *Z-Score (Padronização):* Subtrai a média e divide pelo desvio padrão (deixa tudo centrado no 0).
* *Min-Max (Normalização):* Espreme os valores matematicamente para que fiquem entre 0 e 1.



---

### 1. Aprendizado Supervisionado

Como temos o "gabarito", a matemática foca em reduzir o erro entre a previsão e a resposta real.

**A. Classificação (Categorias)**

* **Cálculos do Motor:**
* *Árvores de Decisão:* Usam **Entropia de Shannon** (nível de bagunça) e **Ganho de Informação** (quanto a entropia cai após um corte). Algumas árvores usam o **Índice de Impureza de Gini** (probabilidade de errar uma classificação aleatória).
* *Naive Bayes:* **Probabilidade Condicional (Teorema de Bayes)**: P(A|B) = P(B|A) * P(A) / P(B). Ele multiplica as probabilidades das características.
* *SVM:* **Multiplicadores de Lagrange** e o **Truque do Kernel** (ex: RBF), que calcula produtos escalares para dobrar o espaço para 3D sem precisar desenhar os pontos fisicamente.
* *Regressão Logística:* **Função Sigmoide** ($1 / (1 + e^{-x})$), que espreme qualquer número em uma probabilidade de 0% a 100%.


* **Cálculos de Avaliação:**
* *Matriz de Confusão:* Conta Verdadeiros/Falsos Positivos e Negativos.
* *Métricas:* **Acurácia** (Acertos gerais), **Precisão** (Dos que eu disse que eram fraude, quantos eram?), **Recall/Sensibilidade** (De todas as fraudes reais, quantas eu achei?), **F1-Score** (Média harmônica entre Precisão e Recall).



**B. Regressão (Números Contínuos)**

* **Cálculos do Motor:**
* *Regressão Linear:* **Mínimos Quadrados Ordinários (OLS)** e **Gradiente Descendente** (cálculo de derivadas para achar o ponto mais baixo de uma curva de erro).


* **Cálculos de Avaliação:**
* *MSE (Erro Quadrático Médio):* Calcula a diferença entre a previsão e o real, eleva ao quadrado e tira a média (penaliza muito os erros grandes).
* *MAE (Erro Absoluto Médio):* Média simples da diferença absoluta.
* *$R^2$ (Coeficiente de Determinação):* Diz qual a porcentagem da variância dos dados o modelo conseguiu explicar (vai de 0 a 1).



---

### 2. Aprendizado Não-Supervisionado

Aqui não há gabarito, então a matemática mede "o quão juntos e o quão separados" os dados estão.

**A. Agrupamento (Clustering)**

* **Cálculos do Motor:**
* *K-Means:* Calcula o **Centroide** (média aritmética das posições de todos os pontos do grupo). A cada passo, tenta minimizar o **SSE (Soma dos Erros Quadráticos)** intra-cluster.
* *Hierárquico:* Usa **Critérios de Linkage** (ligação). *Single-link* calcula a distância entre os pontos mais próximos de dois grupos; *Complete-link* calcula a distância entre os pontos mais distantes.
* *DBSCAN:* Conta matematicamente quantos pontos estão dentro de um raio *Epsilon* ($\epsilon$) para definir se uma área é densa (Core points).


* **Cálculos de Avaliação (Validação Interna):**
* *Índice de Silhueta:* Um valor de -1 a 1 que calcula a **Coesão** (quão perto estou do meu grupo) menos a **Separação** (quão longe estou do grupo vizinho).
* *Coesão (WSS):* Soma das distâncias dentro do grupo.
* *Separação (BSS):* Distância entre os centros dos grupos.



**B. Regras de Associação**

* **Cálculos do Motor & Avaliação (são os mesmos):**
* *Suporte:* Frequência absoluta (Quantas vezes fralda e cerveja aparecem juntas no total das compras?).
* *Confiança:* Probabilidade condicional (Das vezes que compraram fralda, qual a porcentagem que também levou cerveja?).
* *Lift:* Razão estatística que diz se a regra é uma coincidência ou uma associação forte (Lift > 1 significa associação positiva).



---

### 3. Aprendizado Semi-Supervisionado e Ativo

Matemática focada em medir "dúvida" ou transição de rótulos.

* **Cálculos do Motor (Semi-Supervisionado):**
* *Label Propagation:* **Matrizes de Transição de Grafos**. Calcula-se o peso das arestas (distância) entre os nós (dados). A probabilidade de um rótulo "vazar" de um nó para outro é proporcional à equação do Kernel RBF (função de distância exponencial).


* **Cálculos do Motor/Avaliação Interna (Aprendizado Ativo):**
* *Entropia Máxima:* Aplica a fórmula de Shannon. O dado com o maior valor matemático de Entropia é aquele que causa mais confusão (é o que o algoritmo pede para o humano rotular).
* *Margem de Confiança:* $Probabilidade(Classe 1) - Probabilidade(Classe 2)$. O algoritmo pede para rotular o dado cuja subtração dê o valor mais perto de zero (empate).



---

### 4. Aprendizado por Reforço (Reinforcement Learning)

A matemática aqui é voltada para planejamento e otimização temporal.

* **Cálculos do Motor:**
* *Equação de Bellman:* A fórmula que sustenta tudo. Ela diz que o Valor de um estado atual é igual à Recompensa imediata mais o Valor máximo do próximo estado, descontado no tempo.
* *Q-Learning Formula:* Ele atualiza a tabela multiplicando o erro por um **$\alpha$ (Taxa de Aprendizado)** e punindo o futuro com um **$\gamma$ (Fator de Desconto)**, que varia de 0 a 1 (onde valores próximos a 0 fazem a IA ser "imediatista" e próximos a 1 a fazem pensar no "longo prazo").


* **Cálculos de Avaliação:**
* *Recompensa Cumulativa Média:* O total de pontos (ou vitórias) que o agente consegue ao final de um episódio (partida).



---

### 5. Deep Learning (Redes Neurais e IAs Generativas)

Cálculos massivos de Álgebra Linear e Cálculo Diferencial.

* **Cálculos do Motor:**
* *Soma Ponderada e Ativação:* $X \cdot W + B$ (Dados x Pesos + Viés). O resultado passa por uma Função de Ativação (*ReLU*, *Sigmoide*).
* *Backpropagation (Retropropagação):* Usa a **Regra da Cadeia do Cálculo (Derivadas)** para descobrir qual "peso" na rede neural foi culpado pelo erro lá no final.
* *Mecanismo de Atenção (Transformers/LLMs):* Cálculos pesados de Matrizes chamadas *Query (Q)*, *Key (K)* e *Value (V)*. A equação $Softmax((Q \cdot K^T) / \sqrt{d}) \cdot V$ é o que permite ao ChatGPT entender qual palavra importa mais numa frase.


* **Cálculos de Avaliação:**
* *Loss Function (Função de Custo):* *Cross-Entropy Loss* (Mede a distância entre a distribuição de probabilidade da resposta da IA e o gabarito real em classificação) e *MSE* (Para prever valores contínuos).


A fase de preparação dos dados é o verdadeiro "trabalho braçal" da Ciência de Dados e do Machine Learning. Como dita a regra clássica: *Garbage in, Garbage out* (Lixo entra, Lixo sai). Se você entregar dados ruins ou mal formatados para um algoritmo brilhante, ele aprenderá padrões errados.

Com base nos seus materiais (como as aulas de Pré-Processamento e o artigo clássico sobre PCA), vamos destrinchar esses três pilares: a categorização dos atributos, as técnicas de processamento e a magia matemática do PCA.

---

### 1. Categorização de Dados (O que o algoritmo está "lendo"?)

Antes de processar, precisamos entender a natureza de cada coluna do nosso banco de dados. Os algoritmos tratam números e textos de formas completamente diferentes. Os dados se dividem em duas grandes famílias:

* **Qualitativos (Categóricos):** Descrevem qualidades ou características, não quantidades matemáticas.
* **Nominais:** Não existe ordem de grandeza ou hierarquia. Exemplo: *Cor dos olhos (Azul, Castanho, Verde)*, *Sexo*, *Estado Civil*. Não faz sentido dizer que "Azul é maior que Verde".
* **Ordinais:** Existe uma ordem ou ranking claro, mas a distância matemática entre eles não é exata. Exemplo: *Tamanho de roupa (P, M, G, GG)*, *Patente militar*, *Nível de escolaridade*.


* **Quantitativos (Numéricos):** Representam quantidades mensuráveis.
* **Discretos:** Valores inteiros, geralmente resultantes de contagens. Exemplo: *Número de filhos*, *Quantidade de acessos a um site*, *Número de carros na rua*.
* **Contínuos:** Valores reais (com casas decimais), gerados por medições. Exemplo: *Altura (1.75m)*, *Temperatura (23.5ºC)*, *Salário*.



---

### 2. Técnicas de Pré-Processamento

Depois de classificar os dados, precisamos "limpá-los" e ajustá-los. Algoritmos baseados em distância (como KNN e K-Means) ou gradiente (como Redes Neurais) são extremamente sensíveis à escala dos dados.

Aqui estão as principais técnicas:

* **Limpeza (Data Cleansing):** Tratar dados faltantes (ausentes/Nulos) preenchendo-os com a média/mediana ou removendo a linha, e remover *outliers* (erros grotescos de digitação ou sensores quebrados).
* **Transformação de Escala (Crucial):**
* **Normalização (Min-Max):** Espreme todos os valores matematicamente para que fiquem no intervalo exato entre **0 e 1**. Muito útil para Redes Neurais e processamento de imagens.
* **Padronização (Z-Score):** Ajusta os dados para que tenham **Média igual a 0** e **Desvio Padrão igual a 1**. Ao contrário do Min-Max, ela não limita os valores a um máximo absoluto, sendo muito mais resistente a *outliers* na base.


* **Binarização/Encoding:** Algoritmos matemáticos não sabem ler a palavra "Casado" ou "Solteiro". Precisamos transformar as variáveis nominais qualitativas em colunas binárias (0 ou 1) usando técnicas como *One-Hot Encoding*.

---

### 3. PCA (Análise de Componentes Principais)

O PCA (*Principal Component Analysis*) é uma das técnicas mais elegantes de **Redução de Dimensionalidade** (Aprendizado Não-Supervisionado).

Imagine que você tem um banco de dados com 150 colunas (150 dimensões). É impossível visualizar isso graficamente, e os algoritmos ficam lentos e sofrem com a "Maldição da Dimensionalidade".

**Como o PCA resolve isso?**
O PCA condensa as 150 colunas originais em, digamos, 2 ou 3 novas colunas (os *Componentes Principais*), **preservando o máximo de informação (variância) possível**.
O texto que você enviou (`pca.pdf`) usa uma analogia perfeita: o PCA encontra o melhor ângulo para projetar uma "sombra" dos dados. Se você iluminar os dados do ângulo certo, a sombra 2D preserva quase todo o formato do objeto 3D.

**A Matemática por trás:**

1. Ele calcula a matriz de covariância (para ver quais atributos andam juntos).
2. Ele encontra os **Autovetores (Eigenvectors)**: Eles apontam a direção exata no espaço onde os dados estão mais espalhados (maior variância). Esses vetores viram os novos eixos X e Y (os Componentes Principais).
3. Ele encontra os **Autovalores (Eigenvalues)**: São números que dizem o grau de importância de cada Autovetor (ex: "O Componente 1 carrega 80% da informação, o Componente 2 carrega 15%...").

Para que isso fique completamente claro de forma intuitiva e sem equações matemáticas complexas, gerei um simulador visual abaixo. Nele, você poderá tentar achar manualmente o "Componente Principal" de uma nuvem de dados 2D rotacionando um eixo e vendo como a variância (espalhamento) da projeção muda.