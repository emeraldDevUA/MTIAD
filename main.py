import cv2
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


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

# CODE FOR TASK 4
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


def calculate_entropy(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Flatten the image to a 1D array of pixel values
    pixel_values = image.flatten()

    # Get the histogram of pixel values
    histogram, bin_edges = np.histogram(pixel_values, bins=256, range=(0, 256), density=True)

    # Filter out zero probabilities to avoid log(0)
    histogram = histogram[histogram > 0]

    # Compute entropy using the Shannon formula
    entropy = -np.sum(histogram * np.log2(histogram))

    return entropy


def mean_arithmetical_expectation(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixel_values = image.flatten()
    sum_value = np.sum(pixel_values)

    return sum_value / len(pixel_values)


def mean_squared_deviation(image, expectation):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = image.flatten()
    # Calculate the mean squared deviation
    return np.mean(np.square(pixels - expectation))


def get_variable_thresholds(entropies):
    mean_entropy = np.mean(entropies)
    std_entropy = np.std(entropies)

    minus_sigma_value = mean_entropy - 1 * std_entropy

    plus_sigma_value = mean_entropy + 1 * std_entropy
    if minus_sigma_value <= 0:
        minus_sigma_value = plus_sigma_value / 5
    if plus_sigma_value >= np.max(entropies):
        plus_sigma_value = np.max(entropies) * 0.8

    return [minus_sigma_value, plus_sigma_value, np.min(entropies), np.max(entropies)]


def count_distribution(entropies):
    class_a = 0
    class_b = 0
    class_c = 0

    mean_entropy = np.mean(entropies)
    std_entropy = np.std(entropies)

    minus_sigma_value = mean_entropy - 1 * std_entropy
    plus_sigma_value = mean_entropy + 1 * std_entropy

    if minus_sigma_value <= 0:
        minus_sigma_value = plus_sigma_value / 5
    if plus_sigma_value >= np.max(entropies):
        plus_sigma_value = np.max(entropies) * 0.8

    for (value) in entropies:
        if value < minus_sigma_value:
            class_a += 1
        elif value > plus_sigma_value:
            class_c += 1
        else:
            class_b += 1

    return [class_a, class_b, class_c]




def entropy_to_color(entropy, min_entropy, max_entropy, plots=False):
    # Normalize entropy between 0 and 1
    normalized = (entropy - min_entropy) / (max_entropy - min_entropy)

    # Convert to color using a colormap (plt.cm)
    colormap = plt.cm.viridis  # Use 'viridis' or other colormaps like 'plasma', 'coolwarm'
    color = colormap(normalized)  # Returns a tuple (R, G, B, A)

    if plots:
        return color[:3]
    else:
        return tuple([int(255 * c) for c in color[:3]])


# Function to reconstruct the image
def reconstruct_image(entropies, n, image_size, image_name):
    # Create an empty image
    restored_image = Image.new('RGB', image_size)
    draw = ImageDraw.Draw(restored_image)

    try:
        font = ImageFont.truetype("res/Montserrat-Bold.ttf", 50)  # You can adjust the font size
    except IOError:
        font = ImageFont.load_default()

    # Get the minimum and maximum entropy for color scaling
    min_entropy = np.min(entropies)
    max_entropy = np.max(entropies)

    # Number of segments along the width and height
    num_segments_x = image_size[0] // n
    num_segments_y = image_size[1] // n

    # Loop over each segment and paste it onto the restored image
    for i in range(num_segments_y):
        for j in range(num_segments_x):
            # Get the segment index
            idx = i * num_segments_x + j

            # Get the entropy for this segment
            entropy = entropies[idx]

            # Get the color for this entropy value
            color = entropy_to_color(entropy, min_entropy, max_entropy)

            # Draw the n x n block with the corresponding color
            draw.rectangle([j * n, i * n, (j + 1) * n, (i + 1) * n], fill=color)

    text_position = (0, image_size[1] - 100)
    draw.text(text_position, image_name, fill=(0, 0, 0), font=font)
    return restored_image


## Plots
def classification_plot(bars, values, title, xlabel, ylabel, color=None):
    if color is None:
        color = ['#40E0D0', '#D4AF37', '#7f00ff']
    plt.figure(figsize=(8, 6))
    plt.bar(bars,
            values,
            color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()



file_name = 'F-16'
_format = 'bmp'

image = cv2.imread(f'images/{file_name}.{_format}')
height, width, channels = image.shape

# CODE FOR TASK 4
segment_size = 16

segment_array = segment_image_no_overlap(image, segment_size)
segment_entropies = []
mean_sq_dev = []
norm_correlation = []

for i in range(len(segment_array) - 1):
    norm_correlation.append(normalized_correlation(segment_array[i], segment_array[i + 1]))

norm_correlation.append(0)
for (segment) in segment_array:
    segment_entropies.append(calculate_entropy(segment))
    mean_sq_dev.append(mean_squared_deviation(segment, mean_arithmetical_expectation(segment)))




# Plots

# DIAGRAM 1 - 3
entropy_classification = count_distribution(segment_entropies)
mean_sq_dev_classification = count_distribution(mean_sq_dev)
norm_correlation_classification = count_distribution(norm_correlation)

classification_plot(['Distribution 1', 'Distribution 2', 'Distribution 3'], entropy_classification,
                    'Entropy Classification', 'Entropy Class', 'Number of Segments')
classification_plot(['Distribution 1', 'Distribution 2', 'Distribution 3'], mean_sq_dev_classification,
                    'Mean Squared Deviation Classification', 'MSD Class', 'Number of Segments')
classification_plot(['Distribution 1', 'Distribution 2', 'Distribution 3'], norm_correlation_classification,
                    'Normalized Correlation Classification', 'NC Class', 'Number of Segments')

# DIAGRAM 1 - 3

# DIAGRAM 4
entropy_thresholds = get_variable_thresholds(segment_entropies)

color1 = entropy_to_color(entropy_thresholds[0], entropy_thresholds[2], entropy_thresholds[3], True)
color2 = entropy_to_color(entropy_thresholds[1], entropy_thresholds[2], entropy_thresholds[3], True)
entropy_thresholds = [entropy_thresholds[0], entropy_thresholds[1]]

classification_plot(['Threshold 1', 'Threshold 2'], entropy_thresholds, 'Entropy Threshold', 'Entropy Class',
                    'Threshold Value', [color1, color2])
# DIAGRAM 4

# DIAGRAM 5
mean_sq_dev_thresholds = get_variable_thresholds(mean_sq_dev)

color3 = entropy_to_color(mean_sq_dev_thresholds[0], mean_sq_dev_thresholds[2], mean_sq_dev_thresholds[3], True)
color4 = entropy_to_color(mean_sq_dev_thresholds[1], mean_sq_dev_thresholds[2], mean_sq_dev_thresholds[3], True)
mean_sq_dev_thresholds = [mean_sq_dev_thresholds[0], mean_sq_dev_thresholds[1]]

classification_plot(['Threshold 1', 'Threshold 2'], mean_sq_dev_thresholds, 'Mean Squared Deviation Threshold',
                    'MSD Class', 'Threshold Value', color=[color3, color4])
# DIAGRAM 5

# DIAGRAM 6
norm_correlation_thresholds = get_variable_thresholds(norm_correlation)

color5 = entropy_to_color(norm_correlation_thresholds[0], norm_correlation_thresholds[2],
                          norm_correlation_thresholds[3], True)
color6 = entropy_to_color(norm_correlation_thresholds[1], norm_correlation_thresholds[2],
                          norm_correlation_thresholds[3], True)
norm_correlation_thresholds = [norm_correlation_thresholds[0], norm_correlation_thresholds[1]]
classification_plot(['Threshold 1', 'Threshold 2'], norm_correlation_thresholds, 'Normalized Correlation Threshold',
                    'NC Class', 'Threshold Value', color=[color5, color6])
# DIAGRAM 6


entropy_img = reconstruct_image(entropies=segment_entropies, n=segment_size, image_size=(width, height),
                                image_name="Entropy Image Reconstruction")
mean_sq_img = reconstruct_image(entropies=mean_sq_dev, n=segment_size, image_size=(width, height),
                                image_name="MSD Image Reconstruction")
norm_correlation_img = reconstruct_image(entropies=norm_correlation, n=segment_size, image_size=(width, height),
                                         image_name="NC Image Reconstruction")

entropy_img.show("Entropy Image Reconstruction")
mean_sq_img.show("MSD Image Reconstruction")
norm_correlation_img.show("NC Image Reconstruction")

entropy_img.save(f'saves/entropy_{file_name}.bmp')
mean_sq_img.save(f'saves/msd_{file_name}.bmp')
norm_correlation_img.save(f'saves/nc_{file_name}.bmp')
# CODE FOR TASK 4
