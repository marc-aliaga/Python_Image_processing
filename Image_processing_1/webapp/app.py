import io
import os

import requests
from flask import Flask, jsonify, render_template, request, send_file

# URL del servicio en la nube que ejecuta el algoritmo, p. ej. https://mi-api.com/process
ALGORITHM_URL = "http://localhost:8000/process"
ALGORITHM_API_KEY = os.environ.get("ALGORITHM_API_KEY")  # opcional
ALGORITHM_TIMEOUT = float(os.environ.get("ALGORITHM_TIMEOUT", "700"))  # segundos

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB upload limit


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_image():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="No se ha enviado ninguna imagen."), 400
    if not (file.mimetype or "").startswith("image/"):
        return jsonify(error="El archivo no es una imagen válida."), 400
    if not ALGORITHM_URL:
        return jsonify(error="ALGORITHM_URL no está configurada en el servidor."), 500

    headers = {"Authorization": f"Bearer {ALGORITHM_API_KEY}"} if ALGORITHM_API_KEY else {}
    try:
        res = requests.post(
            ALGORITHM_URL,
            files={"image": (file.filename, file.stream, file.mimetype)},
            headers=headers,
            timeout=ALGORITHM_TIMEOUT,
        )
    except requests.RequestException as e:
        app.logger.exception("Remote processing failed")
        return jsonify(error=f"No se pudo contactar con el servicio de procesado: {e}"), 502

    if not res.ok:
        try:
            msg = res.json().get("error") or res.json().get("detail")
        except ValueError:
            msg = None
        return jsonify(error=msg or f"El servicio de procesado devolvió {res.status_code}."), 502

    mimetype = res.headers.get("Content-Type", "image/png")
    return send_file(io.BytesIO(res.content), mimetype=mimetype, download_name="resultado.png")


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="La imagen supera el límite de 15 MB."), 413


if __name__ == "__main__":
    app.run(debug=True)
