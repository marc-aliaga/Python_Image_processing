from skimage import io
import glob
from PIL import Image
from torchvision import transform

path = ""
tensor_list = []


for i in glob.glob(path):
    image = io.imread(i) #this is a numpy array. Dosent have the covnert() method

    ## PIL, image.open()
    transform = transforms.ToTensor()
    image_tensor = image.fromarray(image).convert("RGB")
    image_tensor = transform(image_tensor)

    tensor_list.append(image_tensor)
print("longitud: {len(tensor_list)}")