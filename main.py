import cv2
import numpy as np

import matplotlib.pyplot as plt
import math as m
import NoiseGenerator as generator


def segment_image_no_overlap(image, segment_size):
    """
    Segments the image into equal parts without overlap.

    Parameters:
    - image: Input image
    - segment_size: Size of each square segment (segment_size x segment_size)

    Returns:
    - List of image segments
    """
    height, width, _ = image.shape  # Get the actual dimensions of the image
    segments = []  # Initialize an empty list to store segments

    # Loop through the image to create segments of size `segment_size x segment_size`
    for y in range(0, height, segment_size):
        for x in range(0, width, segment_size):
            segment = image[y:y + segment_size, x:x + segment_size]  # Extract each segment
            if segment.shape[0] == segment_size and segment.shape[1] == segment_size:
                segments.append(segment)  # Add the segment to the list

    return segments  # Return the list of segments


def get_histogram(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    return hist


def plot_single_entropy_chart(entropy_values):
    fig = plt.figure(figsize=(10, 6))

    ax = fig.add_subplot(111)
    x = ['Shannon Entropy', 'Hartley Entropy', 'Partial Shannon', 'Partial Hartley', 'Markov Entropy']
    y = entropy_values

    ax.bar(x, y, color=['blue', 'green', 'orange', 'red', 'purple'])

    ax.set_title('Entropy of the Entire Image and Parts')
    ax.set_xlabel('Entropy Type')
    ax.set_ylabel('Entropy Value')

    plt.tight_layout()
    plt.show()


def count_data_size(img_h, img_w):
    return img_h * img_w * 3 * 8


def count_different_pixels(image1, image2):
    if image1.dtype != image2.dtype:
        image2 = image2.astype(image1.dtype)  #
    # Since the images are guaranteed to be the same size, we can directly compute the difference
    g1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)

    # Convert the difference image to grayscale



    # Count the number of non-zero pixels (different pixels)
    pixels_in_range = np.count_nonzero(diff < 50)

    return pixels_in_range


def mean_sq_deviation(image1, image2):
    # Ensure both images have the same data type
    if image1.dtype != image2.dtype:
        image2 = image2.astype(image1.dtype)

    # Compute the difference between the two images
    diff = cv2.absdiff(image1, image2)

    # Convert the difference to grayscale if necessary (depends on your use case)
    # If you want to compute MSE for color images, you can skip this step.
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Compute the squared differences
    squared_diff = np.square(gray_diff)

    # Compute the mean of the squared differences
    mse = np.mean(squared_diff)

    return mse

def normalized_correlation(image1, image2):
    # Ensure both images have the same size and data type
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions.")
    if image1.dtype != image2.dtype:
        image2 = image2.astype(image1.dtype)

    # Flatten the images into 1D arrays
    img1_flat = image1.flatten()
    img2_flat = image2.flatten()

    # Compute the means of the images
    mean_img1 = np.mean(img1_flat)
    mean_img2 = np.mean(img2_flat)

    # Subtract the mean from the images (center them)
    img1_centered = img1_flat - mean_img1
    img2_centered = img2_flat - mean_img2

    # Calculate the numerator (sum of the element-wise product of the centered images)
    numerator = np.sum(img1_centered * img2_centered)

    # Calculate the denominator (product of the square roots of the sum of squares)
    denominator = np.sqrt(np.sum(img1_centered ** 2) * np.sum(img2_centered ** 2))

    # Calculate the normalized correlation coefficient
    if denominator == 0:
        return 0  # Avoid division by zero
    else:
        correlation_coefficient = numerator / denominator

    return correlation_coefficient


def calculate_psnr(image1, image2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) between two images."""
    mse = mean_sq_deviation(image1, image2)

    # If MSE is zero, the images are identical, and PSNR is infinite
    if mse == 0:
        return float('inf')

    # Maximum possible pixel value of the image
    max_pixel_value = 255.0

    # Compute PSNR
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr
# CODE FOR TASK 3

## Plots


def number_of_erroneous_pixels_plot(erroneous_pixels_gauss, erroneous_pixels_poisson, erroneous_pixels_speckle, pixels):
    plt.figure(figsize=(8, 6))
    plt.bar(['Gauss Noise', 'Poisson Noise', 'Speckle Noise', 'Total Pixels'],
            [erroneous_pixels_gauss, erroneous_pixels_poisson, erroneous_pixels_speckle, pixels],
            color=['blue', 'green', 'orange', 'red'])
    plt.title("Number of Erroneous Pixels")
    plt.xlabel("Noise Type")
    plt.ylabel("Number of Erroneous Pixels")
    plt.tight_layout()
    plt.show()

def error_percentage_per_noise_type_plot(gauss_error, poisson_error, speckle_error):
    plt.figure(figsize=(8, 6))
    plt.bar(['Gauss Noise', 'Poisson Noise', 'Speckle Noise'],
            [gauss_error, poisson_error, speckle_error],
            color=['blue', 'green', 'orange'])
    plt.title("Error Percentage per Noise Type")
    plt.xlabel("Noise Type")
    plt.ylabel("Error Percentage (%)")
    plt.tight_layout()
    plt.show()

def mean_squared_error_per_noise_type_plot(gauss_error_mean, poisson_error_mean, speckle_error_mean):
    plt.figure(figsize=(8, 6))
    plt.bar(['Gauss Noise', 'Poisson Noise', 'Speckle Noise'],
            [gauss_error_mean, poisson_error_mean, speckle_error_mean],
            color=['blue', 'green', 'orange'])
    plt.title("Mean Squared Error per Noise Type")
    plt.xlabel("Noise Type")
    plt.ylabel("Mean Squared Error")
    plt.tight_layout()
    plt.show()

def psnr_per_noise_type_plot(psnr_gauss, psnr_poisson, psnr_speckle):
    plt.figure(figsize=(8, 6))
    plt.bar(['Gauss Noise', 'Poisson Noise', 'Speckle Noise'],
            [psnr_gauss, psnr_poisson, psnr_speckle],
            color=['blue', 'green', 'orange'])
    plt.title("PSNR (Peak Signal-to-Noise Ratio) per Noise Type")
    plt.xlabel("Noise Type")
    plt.ylabel("PSNR (dB)")
    plt.tight_layout()
    plt.show()

def data_transfer_time(transfer_time):
    plt.figure(figsize=(8, 6))
    plt.bar(['Data Transfer'], [transfer_time], color='purple')
    plt.title("Data Transfer Time")
    plt.ylabel("Time (s)")
    plt.tight_layout()
    plt.show()

def normalized_correlation_plot(normalized_correlation_gauss, normalized_correlation_poisson, normalized_correlation_speckle):
    plt.figure(figsize=(8, 6))
    plt.bar(['Gauss Correlation', 'Poisson Correlation', 'Speckle Correlation'],
            [normalized_correlation_gauss, normalized_correlation_poisson, normalized_correlation_speckle],
            color=['blue', 'green', 'orange'])
    plt.title("Normalized Correlation Type")
    plt.xlabel("Noise Type")
    plt.ylabel("Normalized Correlation Value")
    plt.tight_layout()
    plt.show()

## Plots


image = cv2.imread('images/I23.BMP')
height, width, channels = image.shape

distorted_image_gauss = generator.noisy(generator.GAUSS_NOISE, image)
# distorted_image_snp = generator.noisy(generator.SALT_AND_PEPPER, image)
distorted_image_poisson = generator.noisy(generator.POISSON, image)
distorted_image_speckle = generator.noisy(generator.SPECKLE, image)

# Diagram 1
erroneous_pixels_gauss = count_different_pixels(image, distorted_image_gauss)
erroneous_pixels_poisson = count_different_pixels(image, distorted_image_poisson)
erroneous_pixels_speckle = count_different_pixels(image, distorted_image_speckle)

amount_of_pixels = width * height

number_of_erroneous_pixels_plot(erroneous_pixels_gauss, erroneous_pixels_poisson, erroneous_pixels_speckle, amount_of_pixels)
# Diagram 1


# Diagram 2
gauss_error = (erroneous_pixels_gauss / amount_of_pixels)
poisson_error = (erroneous_pixels_poisson / amount_of_pixels)
speckle_error = (erroneous_pixels_speckle / amount_of_pixels)

error_percentage_per_noise_type_plot(gauss_error, poisson_error, speckle_error)

print("Gauss Error ", gauss_error)
print("Poisson Error ", poisson_error)
print("Speckle Error ", speckle_error)
# Diagram 2


#Diagram 3
gauss_error_mean = mean_sq_deviation(image, distorted_image_gauss)
poisson_error_mean = mean_sq_deviation(image, distorted_image_poisson)
speckle_error_mean = mean_sq_deviation(image, distorted_image_speckle)

mean_squared_error_per_noise_type_plot(gauss_error_mean, poisson_error_mean, speckle_error_mean)
#Diagram 3


#Diagram 4
psnr_gauss   = calculate_psnr(image, distorted_image_gauss)
psnr_poisson = calculate_psnr(image, distorted_image_poisson)
psnr_speckle = calculate_psnr(image, distorted_image_speckle)

psnr_per_noise_type_plot(psnr_gauss, psnr_poisson, psnr_speckle)
#Diagram 4


#Diagram 5
net_speed = 50 * pow(10, 6)

data_size = count_data_size(height, width)
transfer_time = data_size / net_speed

data_transfer_time(transfer_time)

print(data_size, "Bytes")
print(transfer_time, "s")
#Diagram 5

#Diagram 6
nc1 = normalized_correlation(image, distorted_image_gauss)
nc2 = normalized_correlation(image, distorted_image_poisson)
nc3 = normalized_correlation(image, distorted_image_speckle)
normalized_correlation_plot(nc1, nc2, nc3)
#Diagram 6


cv2.imshow('gauss img', distorted_image_gauss)
# # cv2.imshow('snp img', distorted_image_snp)
cv2.imshow('poisson img', distorted_image_poisson)
cv2.imshow('speckle img', distorted_image_speckle)

cv2.waitKey(0)

# CODE FOR TASK 3
