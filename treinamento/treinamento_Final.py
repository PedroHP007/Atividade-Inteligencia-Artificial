import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# ==========================================
# 1. CARREGAR DATASET
# ==========================================

df = pd.read_csv("dados/res.csv")


# ==========================================
# 2. SEPARAR CARACTERÍSTICAS E CLASSE
# ==========================================

y = df["class"]

hist_r = df.loc[:, "r0":"r255"].to_numpy()
hist_g = df.loc[:, "g0":"g255"].to_numpy()
hist_b = df.loc[:, "b0":"b255"].to_numpy()


# ==========================================
# 2.1 EXTRAIR ESTATÍSTICAS DA DISTRIBUIÇÃO DE COR
# ==========================================
# Para cada canal (R, G, B), calculamos 5 medidas que descrevem a forma
# da distribuição de cores da imagem, a partir do histograma:
#
#   - media     : tom predominante do canal
#   - desvio    : o quanto a cor varia na imagem (dispersão)
#   - assimetria: se a distribuição "pende" mais para tons claros ou
#                 escuros (skewness)
#   - curtose   : se os valores estão concentrados ou espalhados
#                 (kurtosis)
#   - entropia  : o quão "misturada"/desorganizada é a distribuição de
#                 cores (pode capturar turbidez da água)
#
# No total: 5 medidas x 3 canais = 15 características.

bins = np.arange(256)


def estatisticas_canal(hist, prefixo):
    total = hist.sum(axis=1)
    p = hist / total[:, None]  # distribuição de probabilidade por imagem

    media = (hist * bins).sum(axis=1) / total

    diff = bins[None, :] - media[:, None]

    variancia = (hist * diff ** 2).sum(axis=1) / total
    desvio = np.sqrt(variancia)

    # Evita divisão por zero quando o desvio é 0 (imagem com uma só cor)
    desvio_seguro = np.where(desvio == 0, 1e-8, desvio)

    assimetria = ((hist * diff ** 3).sum(axis=1) / total) / (desvio_seguro ** 3)
    curtose = ((hist * diff ** 4).sum(axis=1) / total) / (desvio_seguro ** 4) - 3

    # Entropia: -soma(p * log2(p)), ignorando bins com probabilidade 0
    p_seguro = np.where(p > 0, p, 1)  # evita log(0); log(1) = 0, não afeta a soma
    entropia = -(p * np.log2(p_seguro)).sum(axis=1)

    return pd.DataFrame({
        f"media_{prefixo}": media,
        f"desvio_{prefixo}": desvio,
        f"assimetria_{prefixo}": assimetria,
        f"curtose_{prefixo}": curtose,
        f"entropia_{prefixo}": entropia,
    })


X = pd.concat([
    estatisticas_canal(hist_r, "r"),
    estatisticas_canal(hist_g, "g"),
    estatisticas_canal(hist_b, "b"),
], axis=1)


# ==========================================
# 3. CRIAR MODELO FINAL
# ==========================================

modelo_final = Pipeline([

    ("scaler", StandardScaler()),

    ("modelo", SVC(kernel="rbf", class_weight="balanced"))

])


# ==========================================
# 4. TREINAR COM 100% DOS DADOS
# ==========================================

modelo_final.fit(X, y)


# ==========================================
# 5. SALVAR MODELO
# ==========================================

joblib.dump(
    modelo_final,
    "modelo/modelo_final.pkl"
)


print("===================================")
print("MODELO FINAL TREINADO")
print("===================================")

print("Algoritmo: SVM")
print("Kernel: RBF")
print("Amostras utilizadas:", len(X))
print("Características:", X.shape[1])
print("Colunas:", X.columns.tolist())

print("\nModelo salvo em:")
print("modelo/modelo_final.pkl")