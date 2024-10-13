import pandas as pd

import yaml
from arquivos_const import ArquivosConsts

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

path = config['processamento']['titanic']['path']

train_df = pd.read_csv(path + ArquivosConsts.TRAIN)
test_df = pd.read_csv(path + ArquivosConsts.TEST)
gender_submission_df = pd.read_csv(path + ArquivosConsts.GENDER_SUBMISSION)

# ETAPA 1

# Explorar os dados
print("Primeiras linhas do train.csv:")
print(train_df.head())

print("\nInformações do train.csv:")
print(train_df.info())

print("\nEstatísticas descritivas do train.csv:")
print(train_df.describe())

# ETAPA 2

# Verificar valores ausentes no train_df
print("Valores ausentes no train.csv:")
print(train_df.isnull().sum())

# Tratar valores ausentes na coluna 'Age' preenchendo com a mediana
mediana_age = train_df['Age'].median()
train_df['Age'].fillna(mediana_age, inplace=True)

# Tratar valores ausentes na coluna 'Embarked' preenchendo com o valor mais frequente
valor_frequente_embarked = train_df['Embarked'].mode()[0]
train_df['Embarked'].fillna(valor_frequente_embarked, inplace=True)

# Tratar valores ausentes na coluna 'Cabin' removendo-a (muitos valores ausentes)
train_df.drop('Cabin', axis=1, inplace=True)

# Verificar novamente valores ausentes após o tratamento
print("\nValores ausentes após o tratamento:")
print(train_df.isnull().sum())

# ETAPA 3

# Selecionar as colunas de interesse
colunas_interesse = ['Pclass', 'Sex', 'Age', 'Fare', 'Survived']
dados_selecionados = train_df[colunas_interesse]

# Converter a coluna 'Sex' em variáveis dummy
dados_selecionados = pd.get_dummies(dados_selecionados, columns=['Sex'], drop_first=True)

print("\nDados após seleção de variáveis e conversão de categorias:")
print(dados_selecionados.head())

# ETAPA 4

# Definir X e y
X = dados_selecionados.drop('Survived', axis=1)
y = dados_selecionados['Survived']

print("\nVariáveis de entrada (X):")
print(X.head())

print("\nVariável de saída (y):")
print(y.head())


#ETAPA 5

# Importando as bibliotecas necessárias
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Supondo que você já tem os dados pré-processados: X e y
# X contém as colunas ['Pclass', 'Sex', 'Age', 'Fare']
# y contém a coluna 'Survived'

# Dividindo os dados em treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criando o modelo k-NN com k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Treinando o modelo
knn.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = knn.predict(X_test)

# Imprimindo as previsões
print("Previsões:", y_pred)

# Avaliando o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")

# Relatório detalhado da classificação (precisão, recall, f1-score)
print("Relatório de classificação:")
print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix

# Calculando a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print("Matriz de Confusão:")
print(cm)


import seaborn as sns
import matplotlib.pyplot as plt

# Plotando a matriz de confusão
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Sobreviveu', 'Sobreviveu'], yticklabels=['Não Sobreviveu', 'Sobreviveu'])
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()
