
from pathlib import Path

import torch
from PIL import Image
from diffusers import AutoPipelineForInpainting

# 1. Model
model_id = "andro-flock/LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Using device: {device}")

# 2. Folders
base_folder = Path(
    r"C:\Users\marca\Desktop\Image_Processing\Image_segmentation\imagenes"
)

mask_folder = base_folder / "masks"
output_folder = base_folder / "results"
output_folder.mkdir(parents=True, exist_ok=True)

if not mask_folder.is_dir():
    raise FileNotFoundError(f"Mask folder not found: {mask_folder}")

# 3. Find original images in the main folder
extensions = {".png", ".jpg", ".jpeg", ".webp"}

original_paths = sorted(
    p for p in base_folder.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
)

if not original_paths:
    raise RuntimeError(f"No original images found in {base_folder}")

print(f"Found {len(original_paths)} original images.")

# 4. Load model
pipe = AutoPipelineForInpainting.from_pretrained(
    model_id,
    torch_dtype=dtype,
    use_safetensors=True,
).to(device)

# 5. Prompts
prompt = (
    "nude woman, completely bare upper torso, "
    "natural bare breasts, realistic natural nipples and areolas, "
    "exact same skin tone as the rest of the body, "
    "matching original skin color perfectly, warm natural skin tone, "
    "realistic skin, natural breast shape, "
    "matching lighting and shadows, seamless photorealistic inpainting, "
    "high detail, sharp focus"
)

negative_prompt = (
    "clothing, shirt, top, bra, lingerie, fabric, straps, "
    "pasties, nipple covers, solid color, "
    "pale skin, white skin, chalky skin, gray skin, ashy skin, "
    "mismatched skin tone, different skin color, "
    "plastic skin, doll-like, unnatural color, "
    "blurry, artifacts, visible seams, halos"
)
# 6. Match each original with its mask
mask_extensions = [".png", ".jpg", ".jpeg", ".webp"]

for index, original_path in enumerate(original_paths, start=1):
    # Find mask with the same filename stem
    mask_path = next(
        (
            mask_folder / f"{original_path.stem}{ext}"
            for ext in mask_extensions
            if (mask_folder / f"{original_path.stem}{ext}").is_file()
        ),
        None,
    )

    if mask_path is None:
        print(f"Skipping {original_path.name}: no matching mask.")
        continue

    print(f"[{index}/{len(original_paths)}] {original_path.name}")

    try:
        with Image.open(original_path) as im:
            image = im.convert("RGB")

        with Image.open(mask_path) as im:
            mask = im.convert("L")

        # Align mask with original image dimensions
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

        # White = regenerate; black = preserve
        mask = mask.point(lambda p: 255 if p >= 128 else 0)

        # SDXL dimensions divisible by 8
        width, height = image.size
        width = max(8, (width // 8) * 8)
        height = max(8, (height // 8) * 8)

        image = image.resize((width, height), Image.Resampling.LANCZOS)
        mask = mask.resize((width, height), Image.Resampling.NEAREST)

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image,
                mask_image=mask,
                width=width,
                height=height,
                num_inference_steps=40,
                guidance_scale=6.5,
                strength=0.7, #
                padding_mask_crop=34,
            ).images[0]

        output_path = output_folder / f"{original_path.stem}_result.png"
        result.save(output_path)
        print(f"Saved: {output_path.name}")

    except Exception as error:
        print(f"Failed on {original_path.name}: {error}")

print(f"Finished. Results saved in: {output_folder}")
