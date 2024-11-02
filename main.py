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
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    segments = []  # Initialize an empty list to store segments

    # Loop through the image to create segments of size `segment_size x segment_size`
    for y in range(0, height, segment_size):
        for x in range(0, width, segment_size):
            segment = image[y:y + segment_size, x:x + segment_size]  # Extract each segment
            if segment.shape[0] == segment_size and segment.shape[1] == segment_size:
                segments.append(segment)  # Add the segment to the list

    return segments  # Return the list of segments


# CODE FOR TASK 4
import numpy as np
from PIL import Image


def normalized_correlation(image1, image2):
    # Convert PIL images to NumPy arrays
    img1_array = np.array(image1, dtype=np.float32)
    img2_array = np.array(image2, dtype=np.float32)

    # Ensure both images have the same size
    # if img1_array.shape != img2_array.shape:
    #     raise ValueError("Images must have the same dimensions.")

    # Flatten the images into 1D arrays
    img1_flat = img1_array.flatten()
    img2_flat = img2_array.flatten()

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
        minus_sigma_value = np.min(entropies) * 1.4
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


from PIL import Image, ImageDraw, ImageFont


def threshold_image(image, values, threshold, n, image_size):
    # Create a new blank image with the same size and mode as the input image
    new_image = Image.new(image.mode, image_size)
    draw = ImageDraw.Draw(new_image)

    try:
        background = image
        new_image.paste(background)
        font = ImageFont.truetype("res/Montserrat-Bold.ttf", 50)  # Adjust the font size as needed
    except IOError:
        font = ImageFont.load_default()

    # Number of segments along the width and height
    num_segments_x = image_size[0] // n
    num_segments_y = image_size[1] // n

    # Loop over each segment and draw rectangles on the new image based on the threshold condition
    for i in range(num_segments_y):
        for j in range(num_segments_x):
            # Get the segment index
            idx = i * num_segments_x + j

            # Get the entropy for this segment
            entropy = values[idx]
            if not threshold[0] < entropy < threshold[1]:
                draw.rectangle([j * n, i * n, (j + 1) * n, (i + 1) * n], fill='Black')

    # Add text to the new image
    text_position = (0, image_size[1] - 100)
    draw.text(text_position, "Thresholded Image", fill='White', font=font)

    return new_image


def calculate_series_lengths(image):
    image_array = np.array(image)

    # Flatten the image array to 1D
    flattened_image = image_array.flatten()

    # Calculate series lengths and count of series
    series_lengths = []
    current_value = flattened_image[0]
    current_length = 1
    series_count = 0

    for i in range(1, len(flattened_image)):
        if flattened_image[i] == current_value:
            current_length += 1
        else:
            # Add the length of the current series
            series_lengths.append(current_length)
            series_count += 1
            # Reset for the new series
            current_value = flattened_image[i]
            current_length = 1

    # Append the final series
    series_lengths.append(current_length)
    series_count += 1

    return [series_count, series_lengths]


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


def count_brightness_surges(segment):
    brightness_surges = 0

    # Convert the input segment to a NumPy array if it isn't already
    image_array = np.array(segment)

    # Assuming the segment is in YCbCr format, extract the Y channel (luminance)
    # The Y channel is typically the first channel in the YCbCr representation
    if image_array.ndim == 3 and image_array.shape[2] == 3:  # Check if it's a 3-channel image
        y_channel = image_array[:, :, 0]  # Get the Y channel
    else:
        # If the input is not in the expected format, raise an error or handle accordingly
        raise ValueError("Input segment must be a 3-channel YCbCr image")

    # Flatten the Y channel to 1D
    flattened_image = y_channel.flatten()

    # Count brightness surges
    for i in range(1, len(flattened_image)):
        if flattened_image[i] != flattened_image[i - 1]:
            brightness_surges += 1

    return brightness_surges


def save_threshold_images(image_name, thresholds, image, n, values, dims):
    img1 = threshold_image(image, values, [thresholds[2], thresholds[0]], n, dims)
    img1.save("saves/0" + image_name + ".png")
    img1 = threshold_image(image, values, [thresholds[0], thresholds[1]], n, dims)
    img1.save("saves/1" + image_name + ".png")
    img1 = threshold_image(image, values, [thresholds[1], thresholds[3]], n, dims)
    img1.save("saves/2" + image_name + ".png")


import csv


def order_mistakes(values, thresholds, file_path, number):
    _list = []

    first_order_mistake = 0
    second_order_mistake = 0

    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            _list.append(row)

    cnt = 0

    correct_values = 0

    for value in values:

        category = 0
        if value < thresholds[0]:
            category = 1
        elif value > thresholds[1]:
            category = 3
        else:
            category = 2

        if category == int(_list[number][cnt]):
            correct_values += 1
        elif category > int(_list[number][cnt]):
            second_order_mistake += 1
        else:
            first_order_mistake += 1

        cnt += 1

    return [correct_values, second_order_mistake, first_order_mistake]


def compute_correlations(series_length, brightness_segments, brightness_surges, entropies, msd):
    # List of features to compare
    features = [series_length, brightness_segments, brightness_surges, entropies, msd]

    # Initialize a 5x5 matrix with zeros
    matrix_size = len(features)
    correlation_matrix = [[0] * matrix_size for _ in range(matrix_size)]

    # Iterate over each pair of features to fill the matrix
    for i in range(matrix_size):
        for j in range(i, matrix_size):
            # Calculate correlation only if both lists have data
            if features[i] and features[j]:
                correlation_value = normalized_correlation(features[i], features[j])
                # Store the correlation in the matrix for both (i, j) and (j, i)
                correlation_matrix[i][j] = correlation_value
                correlation_matrix[j][i] = correlation_value

    return correlation_matrix


file_name = 'F-16'
_format = 'bmp'

image = cv2.imread(f'images/{file_name}.{_format}')
pil_img = Image.open(f'images/{file_name}.{_format}')

height, width, channels = image.shape
image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# CODE FOR TASK 4
segment_size = 128
segment_array = segment_image_no_overlap(image, segment_size)

series_length = []
brightness_segments = []
brightness_surges = []
entropies = []
msd = []
for segment in segment_array:
    tmp = calculate_series_lengths(segment)
    series_length.append(np.sum(tmp[1]) / len(tmp[1]))
    brightness_segments.append(mean_arithmetical_expectation(segment))
    brightness_surges.append(count_brightness_surges(segment))
    entropies.append(calculate_entropy(segment))
    msd.append(mean_squared_deviation(segment, mean_arithmetical_expectation(segment)))

brightness_segments_img = reconstruct_image(brightness_segments, segment_size, (width, height), "Brightness")
# series_img = reconstruct_image(series_count, segment_size, (width, height), "Series count")
series_length_img = reconstruct_image(series_length, segment_size, (width, height), "Series length")
brightness_surges_img = reconstruct_image(brightness_surges, segment_size, (width, height), "Brightness Surges")
entropy_image = reconstruct_image(entropies, segment_size, (width, height), "Entropy image")
msd_image = reconstruct_image(msd, segment_size, (width, height), "MSD")

save_threshold_images('entropy_image', get_variable_thresholds(entropies), pil_img,
                      segment_size, entropies, (width, height))
save_threshold_images('series_length_image', get_variable_thresholds(series_length), pil_img,
                      segment_size, series_length, (width, height))
save_threshold_images('brightness_surges_image', get_variable_thresholds(brightness_surges), pil_img,
                      segment_size, brightness_surges, (width, height))
save_threshold_images('msd_image', get_variable_thresholds(msd), pil_img,
                      segment_size, msd, (width, height))

cv2.imshow("YCbCr Image", image)
series_length_img.show()
brightness_segments_img.show()
brightness_surges_img.show()
entropy_image.show()


def compute_correlations_plot(matrix, titles):
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(len(titles)), titles, rotation=45, ha='right')
    plt.yticks(np.arange(len(titles)), titles)
    plt.title('Correlation Matrix of Image Segment Metrics')

    # Annotate the correlation values on the plot
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            plt.text(j, i, f"{matrix[i][j]:.2f}", ha='center', va='center',
                     color='black' if abs(matrix[i][j]) < 0.7 else 'white')

    plt.tight_layout()
    plt.show()

# series_length, brightness_segments, brightness_surges, entropies, msd
strings = ['Series Length', 'Overall Brightness', 'Brightness Surges', 'Entropy', 'MSD']

matrix = compute_correlations(series_length, brightness_segments, brightness_surges, entropies, msd)

compute_correlations_plot(matrix, strings)

# DIAGRAM 2
series_length_thresholds = get_variable_thresholds(series_length)

first_color = entropy_to_color(series_length_thresholds[0], series_length_thresholds[2], series_length_thresholds[3],
                               True)
second_color = entropy_to_color(series_length_thresholds[1], series_length_thresholds[2], series_length_thresholds[3],
                                True)
series_length_thresholds = [series_length_thresholds[0], series_length_thresholds[1]]
classification_plot(['Threshold 1', 'Threshold 2'], series_length_thresholds, 'Series length Threshold',
                    'NC Class', 'Threshold Value', color=[first_color, second_color])

# DIAGRAM 3
brightness_segments_thresholds = get_variable_thresholds(brightness_segments)

first_color = entropy_to_color(brightness_segments_thresholds[0], brightness_segments_thresholds[2],
                               brightness_segments_thresholds[3], True)
second_color = entropy_to_color(brightness_segments_thresholds[1], brightness_segments_thresholds[2],
                                brightness_segments_thresholds[3], True)

brightness_segments_thresholds = [brightness_segments_thresholds[0], brightness_segments_thresholds[1]]
classification_plot(['Threshold 1', 'Threshold 2'], brightness_segments_thresholds, 'Brightness Threshold',
                    'NC Class', 'Threshold Value', color=[first_color, second_color])

# DIAGRAM 4
brightness_surges_thresholds = get_variable_thresholds(brightness_surges)

first_color = entropy_to_color(brightness_surges_thresholds[0], brightness_surges_thresholds[2],
                               brightness_surges_thresholds[3], True)
second_color = entropy_to_color(brightness_surges_thresholds[1], brightness_surges_thresholds[2],
                                brightness_surges_thresholds[3], True)

brightness_surges_thresholds = [brightness_surges_thresholds[0], brightness_surges_thresholds[1]]
classification_plot(['Threshold 1', 'Threshold 2'], brightness_surges_thresholds, 'Brightness Surges Threshold',
                    'NC Class', 'Threshold Value', color=[first_color, second_color])

entropy_thresholds = get_variable_thresholds(entropies)

first_color = entropy_to_color(entropy_thresholds[0], entropy_thresholds[2], entropy_thresholds[3], True)
second_color = entropy_to_color(entropy_thresholds[1], entropy_thresholds[2], entropy_thresholds[3], True)

entropy_thresholds = [entropy_thresholds[0], entropy_thresholds[1]]
classification_plot(['Threshold 1', 'Threshold 2'], entropy_thresholds, 'Entropy Threshold',
                    'NC Class', 'Threshold Value', color=[first_color, second_color])

file_path = 'expert_estimations/expert1estimate.csv'

print("Entropy:", order_mistakes(entropies, entropy_thresholds, file_path, 0))
# correct, 1-order mistake, 2-order mistake
print("MSE:", order_mistakes(msd, get_variable_thresholds(msd), file_path, 1))
# correct, 1-order mistake, 2-order mistake
print("BrSurges:", order_mistakes(brightness_surges, brightness_surges_thresholds, file_path, 2))
# correct, 1-order mistake, 2-order mistake
print("Series Length:", order_mistakes(series_length, series_length_thresholds, file_path, 3))

cv2.waitKey(0)
