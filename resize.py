from PIL import Image
import os

from resizeimage import resizeimage

for filename in os.listdir('/home/roee/Pictures'):
    if filename.endswith(".jpg"):
        with open(filename, 'r+b') as f:
            with Image.open(f) as image:
                cover = resizeimage.resize_cover(image, [150, 150])
                cover.save(f'resized_{filename}', image.format)
        # print(os.path.join(directory, filename))

