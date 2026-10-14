from skimage import io
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_float


my_image = io.imread("foto2.png")
my_float_image = img_as_float(my_image)
print(my_float_image.min(), my_float_image.max())
print(my_float_image)

"""
random_image = img_as_float(np.random.random([500,500]))
print(random_image.min(), random_image.max())
print(random_image)
plt.imshow()
plt.show()
"""
my_image[10:100, 10:100] = [255, 255 ,0, 20]
plt.imshow(my_image)
plt.show()