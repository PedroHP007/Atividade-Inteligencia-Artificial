from pathlib import Path

import joblib
import numpy as np
from flask import Flask, render_template, request
from PIL import Image

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "modelo" / "best_classification_model.pkl"
SCALER_PATH = BASE_DIR / "modelo" / "minmax_scaler.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "modelo" / "label_encoder.pkl"

modelo = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)


def preprocessar_imagem(imagem):
    """Ajusta a imagem para o mesmo padrão do treinamento do notebook, mas isolando o objeto da imagem real."""
    imagem_rgb = imagem.convert("RGB")

    # Remove o fundo branco/claríssimo para deixar apenas o copo no centro
    arr = np.asarray(imagem_rgb, dtype=np.uint8)
    fundo = np.median(arr[0:10, 0:10], axis=(0, 1))
    dist = np.linalg.norm(arr - fundo, axis=2)
    mascara = dist > 25

    if mascara.any():
        ys, xs = np.where(mascara)
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        margem = 10
        x0 = max(0, x0 - margem)
        y0 = max(0, y0 - margem)
        x1 = min(arr.shape[1], x1 + margem)
        y1 = min(arr.shape[0], y1 + margem)
        imagem_crop = imagem_rgb.crop((x0, y0, x1, y1))
    else:
        imagem_crop = imagem_rgb

    # Normaliza para um quadrado e resize final
    tamanhos = imagem_crop.size
    lado = max(tamanhos)
    canvas = Image.new("RGB", (lado, lado), (255, 255, 255))
    x = (lado - tamanhos[0]) // 2
    y = (lado - tamanhos[1]) // 2
    canvas.paste(imagem_crop, (x, y))
    imagem_final = canvas.resize((16, 16), Image.Resampling.LANCZOS)

    pixels = np.asarray(imagem_final, dtype=np.uint8).reshape(1, -1)
    return scaler.transform(pixels)


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None

    if request.method == "POST":
        arquivo = request.files.get("imagem")

        if arquivo and arquivo.filename:
            imagem = Image.open(arquivo)
            dados = preprocessar_imagem(imagem)
            previsao_codificada = modelo.predict(dados)
            resultado = label_encoder.inverse_transform(previsao_codificada)[0]

    return render_template("index.html", resultado=resultado)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)