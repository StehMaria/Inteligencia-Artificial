<div align=right style=display:inline-block>
<img width=10% src=http://ForTheBadge.com/images/badges/built-with-love.svg>
<img width=15% src=http://ForTheBadge.com/images/badges/made-with-python.svg>
<img width=15% src=https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter>
</div>

<h1 align=center>Redes Neurais Artificiais 🧠</h1> 


### Ferramentas:
* Python 3
* Anaconda
* Jupyter Notebook

**Tier:** Intermediário/Avançado ⭐⭐⭐ 

### Objetivo:
Construir, analisar e comparar modelos de classificação baseados em Redes Neurais em diversos datasets (MNIST, PathMNIST), focando na otimização, avaliação de desempenho e prevenção de overfitting.

### User Story:

-   [ ] Como cientista de dados, quero avaliar o desempenho de diferentes modelos de machine learning utilizando algoritmos de classificação variados e métricas de avaliação adequadas.

### Definition of Done:

-   [ ] Aplicar métricas de avaliação de modelo apropriadas (e.g., acurácia, curvas de perda).
-   [ ] Apresentar boa acurácia nos modelos desenvolvidos, com foco na generalização e prevenção de *overfitting*.
-   [ ] Construir um sistema de reconhecimento e classificação de imagens usando o PathMNIST dataset.
-   [ ] Construir um sistema de reconhecimento e classificação de imagens usando o MNIST dataset.


## 📋 Sumário 

- [Fundamentos Teóricos: O Neurônio Biológico e Artificial](#fundamentos-teóricos-o-neurônio-biológico-e-artificial) 
- [O Neurônio Artificial (Perceptron)](#o-neurônio-artificial-perceptron) 
  - [Combinação Linear e Viés (Bias)](#combinação-linear-e-viés-bias) 
  - [Funções de Ativação](#funções-de-ativação)
- [Redes Multicamadas (Multilayer Perceptrons - MLP)](#redes-multicamadas-multilayer-perceptrons-mlp) 
- [O Processo de Aprendizado: Backpropagation e Gradiente Descendente](#o-processo-de-aprendizado-backpropagation-e-gradiente-descendente) 
  - [Função Custo (Loss Function)](#função-custo-loss-function) 
  - [O Gradiente Descendente (Gradient Descent)](#o-gradiente-descendente-gradient-descent) 
  - [Backpropagation (Retropropagação do Erro)](#backpropagation-retropropagação-do-erro) 
- [Principais Desafios e Ajustes](#principais-desafios-e-ajustes) 
  - [Overfitting e Regularização (Weight Decay)](#overfitting-e-regularização-weight-decay) 
  - [O Problema do Gradiente Desaparecente (Vanishing Gradient)](#o-problema-do-gradiente-desaparecente-vanishing-gradient) 
- [Variações de Arquiteturas de Redes Neurais](#variações-de-arquiteturas-de-redes-neurais)
- [Projeto](#projeto)
  - [Sobre](#sobre)
  - [Como usar](#como-usar)




## 🧠 Fundamentos Teóricos 

As Redes Neurais Artificiais (RNAs) são inspiradas na estrutura do sistema nervoso humano . Elas consistem em estruturas paralelas e massivamente distribuídas compostas por unidades de processamento básicas chamadas **neurônios artificiais**.  

* **Entradas $x$:** O neurônio recebe características ou sinais de entrada representados por $x_1, x_2, \dots, x_m$. Cada entrada representa uma informação diferente sobre o problema que o neurônio está tentando resolver (por exemplo, diferentes atributos de um minério).  

* **Sinapses e Pesos $w$:** Na biologia, as sinapses conectam os neurônios. Na computação, cada sinapse é caracterizada por um **peso sináptico $w$**, que indica a importância ou a força de influência de uma característica (entrada) em relação à decisão do neurônio.   Cada uma dessas conexões de entrada possui um peso específico associado a ela, representados por $w_{k1}, w_{k2}, \dots, w_{km}$.

O peso dita a força de influência ou a importância daquela característica específica.
Se um peso for positivo e alto, a entrada correspondente terá grande influência a favor da ativação do neurônio (conexão excitatória).
Se o peso for negativo, ele agirá contra a ativação (conexão inibitória).
Se o peso for zero (ou muito próximo de zero), aquela entrada específica será praticamente ignorada pelo neurônio.

* **Soma Ponderada:** O neurônio combina os sinais recebidos multiplicando cada entrada pelo seu respectivo peso.  Uma vez que os sinais de entrada passam pelas sinapses e são multiplicados por seus respectivos pesos, o corpo do neurônio artificial precisa combinar essas informações.


## 🤖 O Neurônio Artificial (Perceptron) 

O modelo matemático fundamental de um neurônio artificial (conhecido como Perceptron) realiza duas etapas principais: a combinação linear dos dados e a aplicação de uma função de ativação. 

### Combinação Linear e Viés (Bias) 

A junção de soma combina os sinais realizando a multiplicação de cada entrada pelo seu peso correspondente e somando todos os resultados.
Matematicamente, o potencial de ativação $v_k$ de um neurônio $k$ é dado por: 

$$v_k = \sum_{j=1}^{m} w_{kj} x_j + b_k$$

Onde:  
* $x_j$ são as entradas (features).  
* $w_{kj}$ são os pesos sinápticos entre a entrada $j$ e o neurônio $k$.  
* $m$ é o número total de entradas.  
* $b_k$ é o **Viés (Bias)**.  

Ou seja: $u_k = (x_1 \cdot w_{k1}) + (x_2 \cdot w_{k2}) + \dots + (x_m \cdot w_{km})$.  

Um valor constante chamado Bias é somado a esse resultado $v_k = u_k + b_k$. O bias serve para deslocar o resultado para o lado positivo ou negativo, ajudando a ajustar o comportamento do neurônio.

Pense no bias como a "teimosia" ou a propensão natural do neurônio. Mesmo que todas as entradas venham zeradas, o bias garante que o neurônio ainda possa disparar (ou se manter calado) dependendo de um limiar fixo próprio. Ele funciona como uma linha de translação que desloca o gráfico da função para a esquerda ou direita, facilitando o ajuste do modelo aos dados. 

### Funções de Ativação 
A saída final do neurônio $u_k$ passa por uma função não-linear $f(\cdot)$ chamada de **Função de Ativação**: 

$$u_k = f(v_k) = f\left(\sum_{j=1}^{m} w_{kj} x_j + b_k\right)$$

A função de ativação decide se o neurônio deve "passar a informação adiante" e com qual intensidade. Sem ela, por mais camadas que a rede tivesse, ela se comportaria apenas como uma grande regressão linear simples, incapaz de resolver problemas complexos do mundo real (como reconhecer imagens ou prever textos). As funções de ativação mais comuns incluem: 

* **Degrau (Threshold):** Usada no Perceptron clássico, retorna 0 ou 1 de forma binária.  
* **Sigmoid (Logística):** $f(a) = \frac{1}{1 + e^{-a}}$, comprime valores para o intervalo $(0, 1)$, ideal para probabilidades.  
* **ReLU (Rectified Linear Unit):** $f(a) = \max(0, a)$, muito usada em redes profundas por evitar o apagamento de sinais em valores positivos altos.  
* **Tanh (Tangente Hiperbólica):** $f(a) = \tanh(a)$, mapeia os valores para o intervalo $(-1, 1)$.  

## 🏗️ Redes Multicamadas (Multilayer Perceptrons - MLP) 
Um neurônio sozinho (Perceptron simples) só consegue resolver problemas linearmente separáveis (como a porta lógica E). Para resolver problemas complexos não-lineares, agrupamos vários neurônios em camadas: 

1. **Camada de Entrada (Input Layer):** Recebe os dados brutos do problema (ex: pixels de uma imagem ou variáveis de um cliente).  
2. **Camadas Ocultas (Hidden Layers):** Camadas intermediárias responsáveis por extrair características cada vez mais abstratas e complexas dos dados.  
3. **Camada de Saída (Output Layer):** Produz a predição final da rede (ex: classe da imagem ou valor numérico predito).  

## 🔄 O Processo de Aprendizado: Backpropagation e Gradiente Descendente 

O aprendizado de uma rede neural consiste em ajustar iterativamente os pesos $w$ e os vieses $b$ para minimizar os erros de predição. 

### Função Custo (Loss Function) 
A função custo serve para medir o quão longe a predição da rede neural $\hat{y}$ está do valor real esperado $y$. O tipo de problema determina a escolha da função:  
* Problemas de Regressão: O objetivo é prever um número real contínuo (por exemplo, o preço de uma casa, a temperatura de amanhã ou o teor de um minério). Para isso, a métrica mais comum é o Erro Quadrático Médio (MSE).
* Problemas de Classificação: O objetivo é prever uma categoria ou classe (por exemplo, se um e-mail é "spam" ou "não spam"). Como as saídas costumam ser probabilidades entre 0 e 1, a função mais adequada é a Entropia Cruzada (Cross-Entropy).

**Erro Quadrático (Para uma Amostra)**

A fórmula apresentada para calcular o erro de uma única amostra em problemas de regressão é: 

$$E = \frac{1}{2} (y - \hat{y})^2$$

Onde $y$ é o valor real e $\hat{y}$ é a predição gerada pela rede.  

Para entender essa conta, ela pode ser dividida em três etapas simples:  
A Diferença $(y - \hat{y})$: Calcula-se o erro residual, ou seja, a distância bruta entre o valor real $y$ e o que a rede previu $\hat{y}$. Se a rede previu $8$ mas o real era $10$, a diferença é $2$. Se previu $12$ para o mesmo real $10$, a diferença é $-2$.  
O Quadrado $(\dots)^2$: Eleva-se essa diferença ao quadrado. Isso tem duas funções fundamentais:
* Eliminar os sinais negativos: Tanto um erro de $2$ quanto um erro de $-2$ viram $4$ $2^2 = 4$ e $(-2)^2 = 4$. Isso garante que erros em direções opostas não se cancelem mutuamente.
* Punir erros grandes: Como o valor é elevado ao quadrado, erros pequenos geram penalidades pequenas, mas erros grandes geram penalidades proporcionalmente muito maiores (um erro de tamanho $2$ vira custo $4$, mas um erro de tamanho $10$ vira custo $100$ ).  

A Fração $\frac{1}{2}$: Multiplica-se o resultado por meio. Esse termo é puramente matemático e serve para facilitar o cálculo da derivada da função custo durante o processo de otimização (quando a derivada é calculada, o expoente $2$ "desce" multiplicando e cancela-se com o $\frac{1}{2}$, simplificando a fórmula final).  
Ao estender esse cálculo para todas as amostras do conjunto de dados e tirar a média, obtém-se o Erro Quadrático Médio (MSE).  


**Entropia Cruzada (Cross-Entropy)**

Em problemas de classificação (como o modelo Perceptron discutido no material), o objetivo é fazer com que a rede decida entre classes específicas.

No algoritmo do Perceptron apresentado no texto, a lógica de erro utilizada durante o treinamento para ajustar as conexões segue uma estrutura linear direta: 

$$e_{k}(n) = d_{k}(n) - y_{k}(n)$$

Onde:  

* $d_{k}(n)$: É o valor desejado (desired), ou seja, a classe real correta (por exemplo, classe $0$ ou $1$ ).  
* $y_{k}(n)$: É a saída gerada pela rede neural após passar pela função de ativação.  

Essa conta mede diretamente o desvio na classificação:  

* Se a classe real é $0$ e a rede previu $0$, o erro é $0 - 0 = 0$ (não há alteração nos pesos).  
* Se a classe real é $0$ e a rede previu $1$, o erro é $0 - 1 = -1$, o que dispara o ajuste matemático dos pesos sinápticos para corrigir o modelo.  

### O Gradiente Descendente (Gradient Descent)
O algoritmo do **Gradiente Descendente** guia a atualização dos pesos na direção que reduz o erro. A regra de atualização genérica é: 
 
$$w_{new} = w_{old} - \eta \frac{\partial E}{\partial w}$$

Imagine que você está perdido no topo de uma montanha escura e quer descer até o vale mais fundo (o ponto de erro zero). A taxa de aprendizado $\eta$ representa o tamanho do passo que você dá. O gradiente $\frac{\partial E}{\partial w}$ indica a inclinação do terreno. Se a inclinação for forte, você dá passos maiores; se estiver plano perto do vale, você dá passos bem curtos para não passar direto.  

### Backpropagation (Retropropagação do Erro)
O **Backpropagation** é o algoritmo central que calcula as derivadas parciais (o gradiente) em redes multicamadas utilizando a **Regra da Cadeia** do cálculo. Ele funciona em duas fases:  
1. **Passo Direto (Forward Pass):** Os dados entram na rede, passam pelas camadas ocultas e geram uma predição na saída. O erro é calculado comparando com o gabarito. 
2. **Passo Reverso (Backward Pass):** O erro é propagado de volta da camada de saída em direção à camada de entrada, permitindo que cada neurônio calcule sua responsabilidade no erro e atualize seus pesos usando o gradiente descendente. 

## 🧼 Principais Desafios e Ajustes 
  
### Overfitting e Regularização (Weight Decay) 
Quando a rede neural é complexa demais ou treina por tempo excessivo, ela pode "decorar" os dados de treino em vez de aprender padrões gerais (*overfitting*). Para combater isso, utiliza-se a regularização por penalidade nos pesos, conhecida no contexto de redes neurais como **Weight Decay** (Decaimento de Peso). Adiciona-se um termo à função de custo penalizando pesos muito grandes: 

$$E_{regularizado} = E_{original} + \frac{\lambda}{2} \sum w^2$$
  
O parâmetro $\lambda$ (lambda) controla a importância da regularização. Ao punir pesos excessivamente altos, evitamos que a rede crie curvas extremas e hipersensíveis para explicar ruídos isolados nos dados de treino, forçando-a a manter modelos mais suaves e generalizáveis. 
  
### O Problema do Gradiente Desaparecente (Vanishing Gradient) 
  
Em redes muito profundas com funções de ativação como a Sigmoid, os gradientes calculados nas camadas iniciais podem se tornar infinitamente pequenos durante o Backpropagation $ \approx 0$. Como consequência, as camadas mais próximas da entrada param totalmente de aprender. 
  
* **Solução:** O uso de funções de ativação modernas como a **ReLU** (que mantém o gradiente constante em 1 para entradas positivas) e o uso de inicializações adequadas de pesos mitigam fortemente esse problema.  
  
## 📊 Variações de Arquiteturas de Redes Neurais 
  
Diferentes tipos de problemas computacionais exigem topologias especializadas de redes neurais: 
  
  | Arquitetura | Aplicação Principal | Característica Distintiva | 
  | :--- | :--- | :--- | 
  | **MLP (Multilayer Perceptron)** | Dados tabulares, classificação geral | Conexões totalmente densas (*fully connected*) | 
  | **CNN (Redes Convolucionais)** | Visão computacional, imagens e vídeos | Usa filtros locais e pooling para extração espacial de características | 
  | **RNN / LSTM (Redes Recorrentes)** | Séries temporais, processamento de linguagem natural | Possui conexões de feedback (memória temporal para sequências) | 
  | **Autoencoders** | Redução de dimensionalidade, reconstrução e geração | Rede simétrica que tenta mapear a entrada nela mesma $X \rightarrow \text{Hidden} \rightarrow X$ |


## 🚀 Projeto

### Sobre
Este repositório é dedicado ao estudo e exploração de Redes Neurais Artificiais (RNAs), desde seus fundamentos teóricos até a implementação prática de modelos como o Perceptron e o Multilayer Perceptron (MLP). O objetivo é fornecer exemplos claros e funcionais que demonstrem os conceitos de aprendizado, otimização e avaliação de desempenho em diferentes conjuntos de dados.


| Modelo | Dataset Usado | Papel no Experimento | Métricas de Avaliação | Função de Ativação |
| :--- | :--- | :--- | :--- | :--- |
| **Perceptron** | **MNIST** | **Classificador Linear Simples.** Ótimo para entender a base das redes neurais com dígitos em preto e branco. | Acurácia, Matriz de Confusão | **Step function** (ou Sigmoid/Softmax se adaptado para multi-classe) |
| **Regressão Logística** | **PathMNIST** | **Classificador Linear.** Serve como uma excelente linha de base (*baseline*) para imagens médicas complexas. | Acurácia, F1-Score, AUC-ROC | **Softmax** (Multinomial, já que o PathMNIST possui 9 classes) |
| **MLP** *(Multi-Layer Perceptron)* | **PathMNIST** | **Classificador Não-Linear.** Camadas ocultas aprendem padrões complexos, ideal para tecidos médicos. | Acurácia, F1-Score, AUC-ROC, Loss | **ReLU** (camadas ocultas) e **Softmax** (camada de saída) |


### Como Usar

Para explorar os notebooks e replicar os experimentos, siga os passos abaixo. Tenha o Jupyter Notebook instalado em sua máquina:

1.  Clone o Repositório:
    ```bash
    git clone https://github.com/seu-usuario/Redes-Neurais.git
    cd Redes-Neurais
    ```

2.  Inicie o Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
