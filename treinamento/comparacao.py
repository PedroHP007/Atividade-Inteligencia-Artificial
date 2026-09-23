import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. CARREGAR DATASET
# ==========================================

df = pd.read_csv("dados/res.csv")


# ==========================================
# 2. SEPARAR X E Y
# ==========================================

X = df.drop(columns=["class"])

y = df["class"]


# ==========================================
# 3. DIVIDIR TREINO E TESTE
# ==========================================

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. NORMALIZAÇÃO
# ==========================================

scaler = StandardScaler()

X_treino_escalado = scaler.fit_transform(X_treino)

X_teste_escalado = scaler.transform(X_teste)


# ==========================================
# 5. CRIAR OS MODELOS
# ==========================================

modelos = {

    "Regressão Logística":
        LogisticRegression(max_iter=2000),

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "Árvore de Decisão":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "SVM":
        SVC(kernel="rbf")
}


# ==========================================
# 6. TREINAR E AVALIAR
# ==========================================

for nome, modelo in modelos.items():

    print("\n===================================")
    print(nome)
    print("===================================")

    # Modelos que se beneficiam da normalização
    if nome in ["Regressão Logística", "KNN", "SVM"]:

        modelo.fit(X_treino_escalado, y_treino)

        previsoes = modelo.predict(X_teste_escalado)

    else:

        modelo.fit(X_treino, y_treino)

        previsoes = modelo.predict(X_teste)


    # ==========================================
    # 7. MÉTRICAS
    # ==========================================

    accuracy = accuracy_score(y_teste, previsoes)

    precision = precision_score(
        y_teste,
        previsoes,
        pos_label="sujo",
        zero_division=0
    )

    recall = recall_score(
        y_teste,
        previsoes,
        pos_label="sujo",
        zero_division=0
    )

    f1 = f1_score(
        y_teste,
        previsoes,
        pos_label="sujo",
        zero_division=0
    )


    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))