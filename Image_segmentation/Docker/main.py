import io
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from diffusers import AutoPipelineForInpainting

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEGFORMER_MODEL = "mattmdjaga/segformer_b2_clothes"
INPAINT_MODEL  = "andro-flock/LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING"

PROMPT = (
    "nude woman, completely bare upper torso, "
    "natural bare breasts, realistic natural nipples and areolas, "
    "exact same skin tone as the rest of the body, "
    "matching original skin color perfectly, warm natural skin tone, "
    "realistic skin, natural breast shape, "
    "matching lighting and shadows, seamless photorealistic inpainting, "
    "high detail, sharp focus"
)

NEGATIVE_PROMPT = (
    "clothing, shirt, top, bra, lingerie, fabric, straps, "
    "pasties, nipple covers, solid color, "
    "pale skin, white skin, chalky skin, gray skin, ashy skin, "
    "mismatched skin tone, different skin color, "
    "plastic skin, doll-like, unnatural color, "
    "blurry, artifacts, visible seams, halos"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

# ---------------------------------------------------------------------------
# Global model handles (loaded once at startup)
# ---------------------------------------------------------------------------
_processor = None
_seg_model = None
_pipe = None

app = FastAPI()


@app.on_event("startup")
def startup():
    global _processor, _seg_model, _pipe

    print("Loading SegFormer...")
    _processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL)
    _seg_model = AutoModelForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL)
    _seg_model.eval()
    if device == "cuda":
        _seg_model = _seg_model.to(device)

    print(f"Loading inpainting pipeline on {device}...")
    _pipe = AutoPipelineForInpainting.from_pretrained(
        INPAINT_MODEL,
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)

    print("Models ready.")


def segment_pil(image: Image.Image) -> np.ndarray:
    """Return binary mask (0/255) for class 4 (upper-clothes)."""
    image = image.convert("RGB")
    inputs = _processor(images=image, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _seg_model(**inputs)

    logits = outputs.logits.cpu()
    upsampled = nn.functional.interpolate(
        logits,
        size=image.size[::-1],          # (H, W)
        mode="bilinear",
        align_corners=False,
    )
    prediction = upsampled.argmax(dim=1)[0].numpy()
    return (prediction == 4).astype(np.uint8) * 255


def inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Run SDXL inpainting. White = regenerate, black = keep."""
    # Ensure mask is binary and same size
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    mask = mask.point(lambda p: 255 if p >= 128 else 0)

    # SDXL wants dimensions divisible by 8
    w, h = image.size
    w = max(8, (w // 8) * 8)
    h = max(8, (h // 8) * 8)
    image = image.resize((w, h), Image.Resampling.LANCZOS)
    mask  = mask.resize((w, h), Image.Resampling.NEAREST)

    with torch.inference_mode():
        result = _pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=image,
            mask_image=mask,
            width=w,
            height=h,
            num_inference_steps=40,
            guidance_scale=6.5,
            strength=0.7,
            padding_mask_crop=34,
        ).images[0]

    return result


@app.post("/process")
async def process(image: UploadFile):
    # 1. Read uploaded image
    try:
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida.")

    # 2. Segment → mask
    mask_arr = segment_pil(img)
    mask_img = Image.fromarray(mask_arr)

    # 3. Inpaint
    try:
        result = inpaint(img, mask_img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inpainting failed: {e}")

    # 4. Return final image
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)