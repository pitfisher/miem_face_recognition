import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid

user_name = "aabarstok"
horizontal_positions = ["45_left","center","45_right"]
vertical_positions = ["up","center","down"]

face_images = []
for i in range(9):
    for vertical_position in vertical_positions:
        for horizontal_position in horizontal_positions:
            face_images.append(Image.open(f"./face_dataset/{vertical_position}/{horizontal_position}/{user_name}/glasses.JPG"))

fig = plt.figure(figsize=(1., 1.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, 3),  # creates 3x3 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for ax, im in zip(grid, face_images):
    # Iterating over the grid returns the Axes.
    ax.imshow(im.transpose(Image.ROTATE_90))

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

plt.show()
