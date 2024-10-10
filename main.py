import cv2
import numpy as np
import matplotlib.pyplot as plt
import math as m


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


def get_probs(image, w, h):
    total_cnt = 0
    red_cnt = 0
    blue_cnt = 0
    green_cnt = 0
    yellow_cnt = 0
    white_cnt = 0
    other_color = 0

    for x in range(h):
        for y in range(w):
            total_cnt += 1.0
            b, g, r = image[x, y]
            r = int(r)
            b = int(b)
            g = int(g)
            if r > g + 30 and r > b + 30:
                red_cnt += 1.0
            if b > r + 30 and b > g + 30:
                blue_cnt += 1.0
            if g > r + 30 and g > b + 30:
                green_cnt += 1.0
            if r > 150 and g > 150 and b < 100:
                yellow_cnt += 1.0
            if r > 200 and g > 200 and b > 200:
                white_cnt += 0.1
            else:
                other_color += 1.0
    N = 3
    p1 = round(red_cnt / total_cnt, N)
    p2 = round(blue_cnt / total_cnt, N)
    p3 = round(green_cnt / total_cnt, N)
    p4 = round(yellow_cnt / total_cnt, N)
    p5 = round(white_cnt / total_cnt, N)
    p6 = round(other_color / total_cnt, N)

    return [p1, p2, p3, p4, p5, p6]


def shanon_entropy(probs):
    final_value = 0
    for p in probs:
        if p > 0:
            final_value += p * m.log(p, 2)
    return -final_value


def hartley_entropy(img):
    # Convert image to grayscale (if it's not already)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 'L' mode is for grayscale in PIL

    # Convert image to numpy array
    img_array = np.array(img)

    # Count unique grayscale values
    unique_colors = np.unique(img_array)

    # Number of unique grayscale values
    num_unique_colors = unique_colors.shape[0]

    # Hartley entropy
    if num_unique_colors > 0:
        H0 = m.log2(num_unique_colors)
    else:
        H0 = 0

    return H0


def markov_process(img):
    img_array = np.array(img)

    if len(img_array.shape) == 3:
        img_array = np.mean(img_array, axis=2).astype(int)

    max_value = img_array.max()
    transition_matrix = np.zeros((max_value + 1, max_value + 1))

    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1] - 1):
            current_pixel = img_array[i, j]
            next_pixel = img_array[i, j + 1]
            transition_matrix[current_pixel, next_pixel] += 1

    row_sums = transition_matrix.sum(axis=1)
    if row_sums.all() != 0:
        transition_matrix = transition_matrix / row_sums[:, np.newaxis]

    transition_matrix = np.nan_to_num(transition_matrix)

    return transition_matrix

# First-order Markov Process Calculation
def markov_entropy(image_channel):
    # Create a 256x256 transition matrix (for pixels 0-255)
    transition_matrix = np.zeros((256, 256), dtype=int)

    # Find transitions between pixels
    pixel_values = image_channel.flatten()
    for i in range(len(pixel_values) - 1):
        current_pixel = pixel_values[i]
        next_pixel = pixel_values[i + 1]
        transition_matrix[current_pixel, next_pixel] += 1

    # Normalize the transition matrix
    transition_matrix = transition_matrix / np.sum(transition_matrix)

    # Calculate the entropy for the process
    entropy_value = 0
    for row in transition_matrix:
        for transition in row:
            if transition > 0:
                entropy_value -= transition * np.log2(transition)

    return entropy_value

def calculate_entropy(transition_matrix):
    entropy = 0
    for row in transition_matrix:
        # Remove zero probabilities to avoid log(0)
        non_zero_probs = row[row > 0]
        entropy += -np.sum(non_zero_probs * np.log2(non_zero_probs))

    return entropy

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

def plot_markov_entropy_3d(segments, segment_size):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    entropy_values = []

    for segment in segments:
        entropy_values.append(markov_entropy(segment))

    x_vals = np.arange(0, image.shape[1], segment_size)
    y_vals = np.arange(0, image.shape[0], segment_size)
    x_vals, y_vals = np.meshgrid(x_vals, y_vals)

    x_vals_flat = x_vals.flatten()
    y_vals_flat = y_vals.flatten()

    ax.bar3d(x_vals_flat, y_vals_flat, np.zeros_like(entropy_values), segment_size, segment_size, entropy_values, shade=True, color='purple')

    ax.set_title('Markov Entropy for Image Segments')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Entropy Value')

    plt.show()


segment_size = 64
image = cv2.imread('images/I23.BMP')
height, width, channels = image.shape
segments = segment_image_no_overlap(image, segment_size)

probs = get_probs(image, width, height)
entropy = shanon_entropy(probs)
partial_sh_entropy = 0
partial_hly_entropy = 0
matrix_temp = markov_process(segments[0])
cnt = 0
for i in segments:

    partial_sh_entropy += shanon_entropy(get_probs(i, 64, 64))
    partial_hly_entropy += hartley_entropy(i)

    cnt = cnt + 1

partial_sh_entropy = partial_sh_entropy / (len(segments))
partial_hly_entropy = partial_hly_entropy / len(segments)
matrix_temp = matrix_temp / len(segments)
my_array = [entropy, hartley_entropy(image), partial_sh_entropy, partial_hly_entropy]
hist = get_histogram(image)

transition_matrix = markov_process(image)


entropy_values = [entropy, hartley_entropy(image), partial_sh_entropy, partial_hly_entropy, markov_entropy(image)]

plot_single_entropy_chart(entropy_values)
plot_markov_entropy_3d(segments, segment_size)


