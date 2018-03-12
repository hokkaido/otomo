import torch
import torchvision.transforms as transforms
from PIL import Image

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.ANTIALIAS)
    return img

def save_image(filename, data):
    #img = data.clone().clamp(0, 255).numpy()
    img = data.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

def image_to_tensor(img, img_transforms = []):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose(img_transforms + [
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x.mul(255))
        #normalize
    ])
    return transform(img)

