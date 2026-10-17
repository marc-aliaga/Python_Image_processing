
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn as nn
import glob
import os
import numpy as np


MODEL_NAME = "mattmdjaga/segformer_b2_clothes"
_processor = None
_model = None


def _load_model():
    # Carga el modelo una sola vez y lo reutiliza en las siguientes llamadas
    global _processor, _model
    if _model is None:
        _processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME)
        _model.eval()
    return _processor, _model


def segment_pil(image):
    """Segmenta una imagen PIL y devuelve la máscara como array uint8 (0/255)."""
    processor, model = _load_model()
    image = image.convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    prediction = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    return (prediction == 4).astype(np.uint8) * 255


def segment_image(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    mask_paths = []

    for i in glob.glob(os.path.join(input_dir, "*")):
        if not os.path.isfile(i):
            continue

        try:
            image = Image.open(i).convert("RGB")
        except (OSError, ValueError):
            continue

        print(f"Procesando: {i}")

        mask = segment_pil(image)

        # Guardar la máscara como imagen PNG
        filename = os.path.splitext(os.path.basename(i))[0]
        mask_path = os.path.join(output_dir, f"{filename}.png")

        Image.fromarray(mask).save(mask_path)
        mask_paths.append(mask_path)

        print(f"Máscara guardada en: {mask_path}")

    return mask_paths
