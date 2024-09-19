import cv2
import numpy as np


def greyscale(image, w, h):
    gray_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            b, g, r = image[i, j]
            # Calculate the grayscale value using the formula
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            # Assign the grayscale value to the new image
            gray_image[i, j] = gray_value
    return gray_image


def threshold_processing(image, w, h, threshold):
    th_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if image[i, j] > threshold:
                th_image[i, j] = 0
            else:
                th_image[i, j] = 255

    return th_image

def get_histogram(image, w, h):
    return 0


image = cv2.imread('images/I23.BMP')

height, width, channels = image.shape

gray_image = greyscale(image, width, height)
th_image = threshold_processing(gray_image, width, height, 95)
# cv2.imwrite('gray_image.jpg', gray_image)
# If you want to display it in a window (optional)

cv2.imshow('Grayscale Image', gray_image)
cv2.imshow('Threshold Image', th_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
