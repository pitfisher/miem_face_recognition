from torchvision.utils import make_grid
from torchvision.transforms import transforms, functional
from torchvision import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import os
import tkinter
import numpy as np
import cv2
user_name = "prybakov"
horizontal_positions = ["45_left","center","45_right"]
vertical_positions = ["up","center","down"]

tensors = []

transform = transforms.Compose([
    transforms.ConvertImageDtype(dtype=torch.float),
    transforms.Resize((600,400),antialias=True),
])

for vertical_position in vertical_positions:
    for horizontal_position in horizontal_positions:
        image = os.path.join('./face_dataset', f"{vertical_position}/{horizontal_position}/{user_name}/glasses.JPG")
        transformed_tensor = transform(functional.pil_to_tensor(Image.open(image).transpose(Image.Transpose.ROTATE_90)))
        tensors.append(transformed_tensor)

grid = make_grid(tensors, nrow=3, padding=1)

img = transforms.ToPILImage()(grid)
opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
cv2.namedWindow('title', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('title',cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow('title', opencvImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
