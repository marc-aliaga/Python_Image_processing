
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn as nn
import glob
import os
import numpy as np


def segment_image(input_dir, output_dir):
    # Segmentation - SegFormer
    processor = SegformerImageProcessor.from_pretrained(
        "mattmdjaga/segformer_b2_clothes"
    )
    model = AutoModelForSemanticSegmentation.from_pretrained(
        "mattmdjaga/segformer_b2_clothes"
    )
    model.eval()

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

        prediction = logits.argmax(dim=1)[0].cpu().numpy()
        mask = (prediction == 4).astype(np.uint8) * 255

        # Guardar la máscara como imagen PNG
        filename = os.path.splitext(os.path.basename(i))[0]
        mask_path = os.path.join(output_dir, f"{filename}_mask.png")

        Image.fromarray(mask).save(mask_path)
        mask_paths.append(mask_path)

        print(f"Máscara guardada en: {mask_path}")

    return mask_paths

 
