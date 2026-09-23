import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


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
# 3. CRIAR OS MODELOS
# ==========================================

modelos = {

    "Regressão Logística": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", LogisticRegression(max_iter=2000))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", KNeighborsClassifier(n_neighbors=5))
    ]),

    "Árvore de Decisão":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", SVC(kernel="rbf"))
    ])
}


# ==========================================
# 4. VALIDAÇÃO CRUZADA
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 5. MÉTRICAS
# ==========================================

metricas = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro"
}


# ==========================================
# 6. AVALIAR CADA MODELO
# ==========================================

for nome, modelo in modelos.items():

    resultado = cross_validate(
        modelo,
        X,
        y,
        cv=cv,
        scoring=metricas
    )

    print("\n===================================")
    print(nome)
    print("===================================")

    print(
        "Accuracy :",
        round(resultado["test_accuracy"].mean(), 4)
    )

    print(
        "Precision:",
        round(resultado["test_precision"].mean(), 4)
    )

    print(
        "Recall   :",
        round(resultado["test_recall"].mean(), 4)
    )

    print(
        "F1-score :",
        round(resultado["test_f1"].mean(), 4)
    )