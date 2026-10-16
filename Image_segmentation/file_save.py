import matplotlib.pyplot as plt
from PIL import Image



img1 = Image.open(
    r"C:\Users\marca\Desktop\Image_Processing\Image_segmentation\imagenes\v.png"
)

img2 = Image.open(
    r"C:\Users\marca\Desktop\Image_Processing\Image_segmentation\imagenes\masks\v.png"
)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img1)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(img2, cmap="gray")
axes[1].set_title("Mask")
axes[1].axis("off")

plt.tight_layout()
plt.show()