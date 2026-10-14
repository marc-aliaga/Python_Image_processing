from PIL import Image
import numpy as np

#load image
img = Image.open("foto2.png")
print(type(img)) 

#img.show()
print("pillow format:\n",img.format)

#convert pillow to array
img1 = np.asarray(img)
print("Pillow asarray format:\n",type(img1))

####################################

#Matplotlib
import matplotlib.image as mpimg # alias cretion mpimg == matplotlib.Image
import matplotlib.pyplot as plt

img = mpimg.imread("foto2.png")
print("Matplotlib fomrat:\n", type(img))
print(img.shape)
plt.imshow(img)
plt.colorbar()


print(img[0, 0])
print(img[0, 0, 0]) #red channel pixel 1
print(img[0, 0, 1]) #green channel pixel 1
print(img[0, 0, 2]) #blue channel pixel 1

############################################

from skimage import io, color, feature, img_as_float

#edge detection 
image = io.imread("foto2.png")
print(image.shape)
rgbimage = image[:, :,:3]
print(rgbimage.shape)
gray = color.rgb2gray(rgbimage)

edges = feature.canny(gray)
print(edges.dtype) # bool data type


#pixel 
pixel_int = image[0,0]
print(pixel_int)
print(img_as_float(pixel_int))
pixel_float = np.array(pixel_int, dtype=np.float32)
print(pixel_float)

#matplotlib
plt.imshow(edges, cmap="gray")
plt.axis("off")
plt.show()


#### opencv-python
import cv2
gray_img = cv2.imread("foto2.png",cv2.IMREAD_COLOR) # IMREAD_COLOR == 1, 0 == GRAY
cv2.imshow("Image",gray_img)
cv2.waitKey()