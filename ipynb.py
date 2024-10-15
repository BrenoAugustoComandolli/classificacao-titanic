from operacoes.titanic_data_base_manager import TitanicDataBaseManager
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ETAPA - Pré processamento

# 1. Carregar a Base de Dados
data_base = TitanicDataBaseManager()

print("\n#########################################")
print("Etapa 1")
print("#########################################\n")
data_base.exibirDados(data_base.get_train())
#data_base.exibirDados(data_base.get_gender_submission())
#data_base.exibirDados(data_base.get_test())

# 2.1 Verificar valores ausentes
print("\n----------------------------------------------------------------------------------")
print("Valores ausentes:")
print(data_base.get_train().isnull().sum())
print("----------------------------------------------------------------------------------\n")

# 2.2 Tratar valores ausentes nas colunas
mediana_age = data_base.get_train()['Age'].median()
data_base.get_train()['Age'].fillna(mediana_age, inplace=True)

moda_embarked = data_base.get_train()['Embarked'].mode()[0]
data_base.get_train()['Embarked'].fillna(moda_embarked, inplace=True)

data_base.get_train().dropna(subset=['Cabin'], inplace=True)

# 3.1 Exibir estatísticas de valores nulos
print("\n----------------------------------------------------------------------------------")
print("Valores ausentes após o tratamento:")
print(data_base.get_train().isnull().sum())
print("----------------------------------------------------------------------------------\n")

# ETAPA 3 - Seleção de variáveis

# 3.2 Selecionar as colunas de interesse
colunas_interesse = ['Pclass', 'Sex', 'Age', 'Fare', 'Survived']
dados_selecionados = data_base.get_train()[colunas_interesse]

# 3.3 Converter a coluna 'Sex' em variáveis dummy
dados_selecionados = pd.get_dummies(dados_selecionados, columns=['Sex'], drop_first=True)

# 3.4 Exibe conversão
print("\n----------------------------------------------------------------------------------")
print("\nDados após seleção de variáveis e conversão de categorias:")
print("----------------------------------------------------------------------------------\n")
print(dados_selecionados.head())

# ETAPA 4 - Divisão de dados

# 4.1 Definir X e y
X = dados_selecionados.drop('Survived', axis=1)
y = dados_selecionados['Survived']

# 4.2 Definir X e y
print("\n----------------------------------------------------------------------------------")
print("Variáveis de entrada (X):")
print("----------------------------------------------------------------------------------\n")
print(X.head())

# 4.3 Definir X e y
print("\n----------------------------------------------------------------------------------")
print("Variável de saída (y):")
print("----------------------------------------------------------------------------------\n")
print(y.head())

# ETAPA 5 - Importar e Treinar o Modelo

# Supondo que você já tem os dados pré-processados: X e y
# X contém as colunas ['Pclass', 'Sex', 'Age', 'Fare']
# y contém a coluna 'Survived'

# 5.1 Dividindo os dados em treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5.2 Criando o modelo k-NN com k=3
knn = KNeighborsClassifier(n_neighbors=3)

# 5.3 Treinando o modelo

print("X_train", X_train)
print("y_train", X_train)
knn.fit(X_train, y_train)

# 5.4 Fazendo previsões no conjunto de teste
y_pred = knn.predict(X_test)

# 5.5 Imprimindo as previsões
print("\n----------------------------------------------------------------------------------")
print("Previsões:", y_pred)
print("----------------------------------------------------------------------------------\n")

# ETAPA 6

# 6.1 Avaliando o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print("\n----------------------------------------------------------------------------------")
print(f"Acurácia do modelo: {accuracy:.2f}")
print("----------------------------------------------------------------------------------\n")

# 6.2 Relatório detalhado da classificação (precisão, recall, f1-score)
print("Relatório de classificação:")
print("\n----------------------------------------------------------------------------------")
print(classification_report(y_test, y_pred))
print("----------------------------------------------------------------------------------\n")

# 6.3 Calculando a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print("\n----------------------------------------------------------------------------------")
print("Matriz de Confusão:")
print(cm)
print("----------------------------------------------------------------------------------\n")

# 6.4 Plotando a matriz de confusão
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Sobreviveu', 'Sobreviveu'], yticklabels=['Não Sobreviveu', 'Sobreviveu'])
plt.xlabel('Previsão')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()
