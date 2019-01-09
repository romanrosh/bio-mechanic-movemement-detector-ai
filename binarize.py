# Load image
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as greyscale
image_grey = cv2.imread('snatch2.jpg', cv2.IMREAD_GRAYSCALE)

# Apply adaptive thresholding
max_output_value = 255
neighorhood_size = 99
subtract_from_mean = 10
image_binarized = cv2.adaptiveThreshold(image_grey,
                                        max_output_value,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        neighorhood_size,
                                        subtract_from_mean)

# Show image
plt.imshow(image_binarized, cmap='gray'), plt.axis("off")
plt.show()