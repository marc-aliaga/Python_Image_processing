#pip install czifile

#import czifile
#img = czifile.imread("test.czi")
#print(img.shape)


####global

import cv2
import glob

path = "C:/Users/marca/Desktop/Projects/Undressed/data/images/*"

for i in glob.glob(path):
    print(i)