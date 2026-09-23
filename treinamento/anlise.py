import pandas as pd

# Carregar o dataset CSV
df = pd.read_csv("dados/res.csv")

# Mostrar as primeiras linhas
print("\nPrimeiras linhas:")
print(df.head())

# Mostrar quantidade de linhas e colunas
print("\nTamanho do dataset:")
print(df.shape)

# Mostrar nomes das colunas
print("\nColunas:")
print(df.columns.tolist())

# Mostrar quantidade de cada classe
print("\nQuantidade por classe:")
print(df["class"].value_counts())

# Verificar valores ausentes
print("\nValores ausentes:")
print(df.isnull().sum().sum())