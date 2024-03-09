# HOWTO: https://dev.to/neeldev96/generate-an-image-grid-using-python-pytorch-97
from torchvision.utils import make_grid, save_image
from torchvision.transforms import transforms, functional
from torchvision import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import os
import tkinter
import numpy as np
import cv2
from tqdm import tqdm

dataset_dir = r'C:\Users\HYPERPC\Documents\miem-project-1712\upd_face_dataset'
# user_name = "prybakov"
# user_name = "aabarstok"
horizontal_positions = ["45_left","center","45_right"]
vertical_positions = ["up","center","down"]
users_list = os.listdir(r'C:\Users\HYPERPC\Documents\miem-project-1712\upd_face_dataset\center\center')
save_dir = os.path.join(dataset_dir, "grids")


# check if saving directory does exist
isExist = os.path.exists(save_dir)
if not isExist:
    os.makedirs(save_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ConvertImageDtype(dtype=torch.float),
    transforms.Resize((600,400),antialias=True),
])

def save_grid_image(tensors, glasses):
    grid = make_grid(tensors, nrow=3, padding=1)
    img = transforms.ToPILImage()(grid)
    # img.show()
    hpercent = (1080/float(img.size[1]))
    wsize = int((float(img.size[0]) * float(hpercent)))
    img = img.resize((wsize,1080), Image.LANCZOS)
    save_image(grid, os.path.join(save_dir, f"{user_name}_grid_{glasses}.jpg"))
    
for user_name in tqdm(users_list):
    tensors_glasses = []
    tensors_no_glasses = []
    image_glasses = os.path.join(dataset_dir, f"{vertical_positions[0]}/{horizontal_positions[0]}/{user_name}/glasses.JPG")
    image_no_glasses = os.path.join(dataset_dir, f"{vertical_positions[0]}/{horizontal_positions[0]}/{user_name}/no_glasses.JPG")
    yesGlasses =  os.path.exists(image_glasses)
    noGlasses = os.path.exists(image_no_glasses)
    if noGlasses:
        if (os.path.exists(os.path.join(save_dir, f"{user_name}_grid_no_glasses.jpg"))):
            continue
        for vertical_position in vertical_positions:
            for horizontal_position in horizontal_positions:
                image_no_glasses = os.path.join(dataset_dir, f"{vertical_position}/{horizontal_position}/{user_name}/no_glasses.JPG")      
                # PIL respects image metadata and since our camera was rotated during shoot all images are opened rotated and have to be rotated back
                transformed_tensor = transform(functional.pil_to_tensor(Image.open(image_no_glasses).transpose(Image.ROTATE_90)))
                # transformed_tensor = transform(functional.pil_to_tensor(Image.open(image)))
                tensors_no_glasses.append(transformed_tensor)
        save_grid_image(tensors_no_glasses, "no_glasses")
    else: print(f"No no_glasses images for {user_name}")
    if yesGlasses:
        if (os.path.exists(os.path.join(save_dir, f"{user_name}_grid_glasses.jpg"))):
            continue
        for vertical_position in vertical_positions:
            for horizontal_position in horizontal_positions:
                image_glasses = os.path.join(dataset_dir, f"{vertical_position}/{horizontal_position}/{user_name}/glasses.JPG")
                # PIL respects image metadata and since our camera was rotated during shoot all images are opened rotated and have to be rotated back
                transformed_tensor = transform(functional.pil_to_tensor(Image.open(image_glasses).transpose(Image.ROTATE_90)))
                # transformed_tensor = transform(functional.pil_to_tensor(Image.open(image)))
                tensors_glasses.append(transformed_tensor)
        save_grid_image(tensors_glasses, "glasses")
    else: print(f"No glasses images for {user_name}")

# opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
# # cv2.namedWindow('title', cv2.WINDOW_KEEPRATIO)
# # cv2.setWindowProperty('title', cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
# # cv2.setWindowProperty('title', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# cv2.imshow('title', opencvImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
