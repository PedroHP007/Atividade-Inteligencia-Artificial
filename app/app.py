from flask import Flask, render_template, request
import joblib
import numpy as np
from PIL import Image

app = Flask(__name__)

# Carrega o modelo que treinamos anteriormente
modelo = joblib.load("../modelo/modelo_final.pkl")


def estatisticas_canal(canal):
    """
    Calcula, a partir dos pixels de um canal de cor (matriz 2D com
    valores 0-255), as 5 estatísticas usadas no treinamento:
    média, desvio padrão, assimetria, curtose e entropia.

    IMPORTANTE: a ordem e a forma de cálculo têm que ser IDÊNTICAS às
    usadas no treinamento_Final.py, senão o modelo recebe dados fora
    do padrão que aprendeu.
    """
    valores = canal.flatten().astype(np.float64)

    media = valores.mean()
    desvio = valores.std()
    desvio_seguro = desvio if desvio != 0 else 1e-8

    diff = valores - media
    assimetria = np.mean(diff ** 3) / (desvio_seguro ** 3)
    curtose = np.mean(diff ** 4) / (desvio_seguro ** 4) - 3

    # Entropia calculada a partir do histograma de 256 bins do canal
    hist, _ = np.histogram(valores, bins=256, range=(0, 256))
    total = hist.sum()
    p = hist / total
    p_seguro = np.where(p > 0, p, 1)
    entropia = -(p * np.log2(p_seguro)).sum()

    return media, desvio, assimetria, curtose, entropia


def extrair_caracteristicas(caminho_imagem):
    # Abre a imagem e garante que ela esteja no formato RGB
    imagem = Image.open(caminho_imagem).convert("RGB")

    # Converte a imagem para um array do NumPy
    imagem = np.array(imagem)

    r_stats = estatisticas_canal(imagem[:, :, 0])
    g_stats = estatisticas_canal(imagem[:, :, 1])
    b_stats = estatisticas_canal(imagem[:, :, 2])

    # A ordem tem que ser IDÊNTICA à do treinamento_Final.py:
    # media_r, desvio_r, assimetria_r, curtose_r, entropia_r,
    # media_g, desvio_g, assimetria_g, curtose_g, entropia_g,
    # media_b, desvio_b, assimetria_b, curtose_b, entropia_b
    caracteristicas = np.array([
        *r_stats,
        *g_stats,
        *b_stats,
    ])

    return caracteristicas


@app.route("/", methods=["GET", "POST"])
def index():

    resultado = None

    if request.method == "POST":

        arquivo = request.files["imagem"]

        if arquivo:

            caracteristicas = extrair_caracteristicas(
                arquivo
            )

            # Transforma em uma linha com 15 características
            caracteristicas = caracteristicas.reshape(1, -1)

            # Faz a previsão
            previsao = modelo.predict(caracteristicas)

            resultado = previsao[0]

    return render_template(
        "index.html",
        resultado=resultado
    )


if __name__ == "__main__":
    app.run(debug=True)