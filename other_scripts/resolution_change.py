import os
import glob
from PIL import Image

input_folder = "C:\\Users\\milen\\OneDrive\\Рабочий стол\\conv_faces"
output_folder = "C:\\Users\\milen\\OneDrive\\Рабочий стол\\new_faces"

os.makedirs(output_folder, exist_ok=True)

resolutions = [1024, 768, 512, 256, 224, 128, 112, 96, 64, 32]

files = glob.glob(os.path.join(input_folder, '*.jpg'))

for f in files:
    img = Image.open(f)
    base_name = os.path.basename(f)
    name, ext = os.path.splitext(base_name)

    image_folder = os.path.join(output_folder, name)
    os.makedirs(image_folder, exist_ok=True)

    original_folder = os.path.join(image_folder, "original")
    os.makedirs(original_folder, exist_ok=True)
    original_path = os.path.join(original_folder, base_name)
    img.save(original_path)

    for resolution in resolutions:
        img_resized = img.resize((resolution, resolution))

        resolution_folder = os.path.join(image_folder, f"{resolution}x{resolution}")
        os.makedirs(resolution_folder, exist_ok=True)

        new_name = f"{name}_{resolution}x{resolution}{ext}"
        dst_path = os.path.join(resolution_folder, new_name)
        img_resized.save(dst_path)

        print(f"{base_name} изменен до {resolution}x{resolution} и сохранен в {dst_path}")
