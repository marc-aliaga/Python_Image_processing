import io

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from segformer import _load_model, segment_pil

app = FastAPI()


@app.on_event("startup")
def startup():
    # Cargar el modelo al arrancar para que la primera petición no tarde
    _load_model()


@app.post("/process")
async def main(image: UploadFile):
    # image es el archivo que el usuario ha subido en la web (campo "image" del POST)
    try:
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida.")

    mask = segment_pil(img)

    # Devolver la máscara como PNG a la web
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
