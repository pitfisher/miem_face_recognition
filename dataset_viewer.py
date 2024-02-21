import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

user_name = "aabarstok"
horizontal_positions = ["45_left","center","45_right"]
vertical_positions = ["up","center","down"]

face_images = []
for i in range(9):
    for vertical_position in vertical_positions:
        for horizontal_position in horizontal_positions:
            face_images.append(Image.open(f"./face_dataset/{vertical_position}/{horizontal_position}/{user_name}/glasses.JPG"))

fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)

for i in range(3):
    for j in range(3):
        ax[i,j].imshow(face_images[i+j].transpose(Image.ROTATE_90))

plt.show()