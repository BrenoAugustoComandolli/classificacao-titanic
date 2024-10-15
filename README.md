# Análise de Desempenho do Modelo K-NN no Conjunto de Dados Titanic

## Descrição do Projeto

Este projeto utiliza um classificador k-NN (k-Nearest Neighbors) para prever a sobrevivência de passageiros do Titanic com base em características como classe, sexo, idade e tarifa paga.

## Etapas do Projeto

1. **Carregamento da Base de Dados**: A base de dados Titanic foi carregada e explorada.
2. **Tratamento de Dados Ausentes**: Valores ausentes foram tratados utilizando a mediana para a idade e a moda para o porto de embarque. Registros com valores ausentes na coluna 'Cabin' foram removidos.
3. **Seleção de Variáveis**: Foram selecionadas as colunas relevantes para a análise: `Pclass`, `Sex`, `Age`, `Fare` e `Survived`. A coluna `Sex` foi convertida em variáveis dummy para permitir a análise.
4. **Divisão dos Dados**: Os dados foram divididos em conjuntos de treino (70%) e teste (30%).
5. **Treinamento do Modelo**: Um modelo k-NN foi treinado utilizando k=3.
6. **Avaliação do Modelo**: O desempenho do modelo foi avaliado com base na acurácia, relatório de classificação e matriz de confusão.

## Desempenho do Modelo

### Acurácia

A acurácia do modelo foi de **X.XX**, indicando a porcentagem de previsões corretas em relação ao total de previsões feitas.

### Relatório de Classificação

O relatório de classificação inclui métricas como precisão, recall e F1-score. A precisão representa a proporção de verdadeiros positivos em relação ao total de positivos previstos, enquanto o recall indica a capacidade do modelo de identificar todos os verdadeiros positivos. O F1-score é a média harmônica entre precisão e recall, proporcionando uma medida balanceada.

```plaintext
Relatório de classificação:
               precision    recall  f1-score   support

           0       X.XX      X.XX      X.XX       XX
           1       X.XX      X.XX      X.XX       XX

    accuracy                           X.XX       XX
   macro avg       X.XX      X.XX      X.XX       XX
weighted avg       X.XX      X.XX      X.XX       XX
```

### Matriz de Confusão

A matriz de confusão fornece uma visão detalhada sobre as previsões do modelo. Os valores na matriz representam:

- **Verdadeiros Negativos (TN)**: Passageiros que não sobreviveram e foram corretamente classificados.
- **Falsos Positivos (FP)**: Passageiros que não sobreviveram, mas foram incorretamente classificados como sobreviventes.
- **Falsos Negativos (FN)**: Passageiros que sobreviveram, mas foram incorretamente classificados como não sobreviventes.
- **Verdadeiros Positivos (TP)**: Passageiros que sobreviveram e foram corretamente classificados.

Matriz de Confusão:
[[TN  FP]
 [FN  TP]]

### Análise e Melhorias
A partir das métricas apresentadas, o modelo apresenta um desempenho razoável, mas há espaço para melhorias. Algumas sugestões incluem:

Testar diferentes valores de "k": O valor de k pode influenciar significativamente o desempenho do modelo. É recomendável realizar uma validação cruzada para encontrar o valor ideal.
Explorar mais características: A inclusão de mais variáveis relevantes pode ajudar a melhorar a precisão do modelo.
Ajustar os parâmetros do modelo: Realizar uma busca em grade para otimizar os hiperparâmetros do k-NN.

### Conclusão
O modelo k-NN é uma abordagem eficaz para prever a sobrevivência dos passageiros do Titanic. A análise das métricas de desempenho e da matriz de confusão fornece insights valiosos sobre a eficácia do modelo e possíveis áreas de melhoria.