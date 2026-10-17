
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

# A tiny 2 x 2 RGB image
array = np.array([
    [[255, 0, 0], [0, 255, 0]],
    [[0, 0, 255], [128, 128, 128]]
], dtype=np.uint8)


plt.imshow(array)
plt.axis("off")
plt.show()


print("NumPy shape:", array.shape)
print("NumPy dtype:", array.dtype)
print("Original pixel:", array[0, 0])

pil_image = Image.fromarray(array)

tensor = transforms.ToTensor()(pil_image)

print("Tensor shape:", tensor.shape)
print("Tensor dtype:", tensor.dtype)
print("Red channel at pixel (0, 0):", tensor[0, 0, 0])
print("Green channel at pixel (0, 0):", tensor[1, 0, 0])
print("Blue channel at pixel (0, 0):", tensor[2, 0, 0])
