# Load image
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image as greyscale
image_gray = cv2.imread('snatch3.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate median intensity
median_intensity = np.median(image_gray)

# Set thresholds to be one standard deviation above and below median intensity
lower_threshold = int(max(0, (1.0) * median_intensity))
upper_threshold = int(min(255, (5.0) * median_intensity))

# Apply canny edge detector
image_canny = cv2.Canny(image_gray, lower_threshold, upper_threshold)

# Show image
plt.imshow(image_canny, cmap='gray'), plt.axis("off")
plt.show()