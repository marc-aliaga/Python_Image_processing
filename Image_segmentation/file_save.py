
from transformers import (
    SegformerImageProcessor,
    AutoModelForSemanticSegmentation,
)
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# 1. Cargar el procesador y el modelo una sola vez
processor = SegformerImageProcessor.from_pretrained(
    "mattmdjaga/segformer_b2_clothes"
)
model = AutoModelForSemanticSegmentation.from_pretrained(
    "mattmdjaga/segformer_b2_clothes"
)
model.eval()

# 2. Rutas
path = r"C:\Users\marca\Desktop\Image_Processing\Image_segmentation"
output_dir = os.path.join(path, "masks")
os.makedirs(output_dir, exist_ok=True)

# Buscar imágenes con distintas extensiones
image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    image_paths.extend(glob.glob(os.path.join(path, ext)))

print(f"Imágenes encontradas: {len(image_paths)}")

# 3. Procesar cada imagen
for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # Ajustar los logits al tamaño original
    logits = F.interpolate(
        outputs.logits,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )

    # Clase predicha para cada píxel
    mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    # 4. Guardar la máscara como imagen PNG
    filename = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(output_dir, f"{filename}_mask.png")

    Image.fromarray(mask).save(mask_path)

    # 5. Mostrar la imagen original y su máscara
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image)
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    axes[1].imshow(
        mask,
        cmap="nipy_spectral",
        vmin=0,
        vmax=model.config.num_labels - 1,
        interpolation="nearest",
    )
    axes[1].set_title("Máscara de segmentación")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    print(f"Máscara guardada: {mask_path}")

print("Segmentación completada.")
