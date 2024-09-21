import cv2
import numpy as np
import matplotlib.pyplot as plt

def greyscale(image, w, h):
    gray_image = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            b, g, r = image[i, j]
            # Calculate the grayscale value using the formula
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            # Assign the grayscale value to the new image
            gray_image[i, j] = gray_value
    return gray_image

def threshold_processing(image, w, h, threshold):
    th_image = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if image[i, j] > threshold:
                th_image[i, j] = 0
            else:
                th_image[i, j] = 255
    return th_image

def get_histogram(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    return hist

image = cv2.imread('images/I23.BMP')

height, width, channels = image.shape

gray_image = greyscale(image, width, height)
th_image = threshold_processing(gray_image, width, height, 95)

hist_original = get_histogram(gray_image)
hist_threshold = get_histogram(th_image)

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

axs[0, 1].imshow(th_image, cmap='gray')
axs[0, 1].set_title('Threshold Image')
axs[0, 1].axis('off')

axs[1, 0].plot(hist_original, color='black')
axs[1, 0].set_title('Histogram (Original)')

axs[1, 1].plot(hist_threshold, color='black')
axs[1, 1].set_title('Histogram (Threshold)')

plt.tight_layout()
plt.show()
