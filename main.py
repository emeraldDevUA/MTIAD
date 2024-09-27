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
    height, width = image.shape  # Get the dimensions of the image
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


def get_entropy(probs):
    final_value = 0
    for p in probs:
        if p > 0:
            final_value += p * m.log(p, 2)
    return -final_value


def hartley_entropy(img):
    # Перетворюємо зображення на масив numpy
    img_array = np.array(img)

    # Рахуємо кількість унікальних кольорів у зображенні
    unique_colors = np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0)

    # Кількість унікальних кольорів
    num_unique_colors = unique_colors.shape[0]

    # Міра Хартлі
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


def plot_combined_3d(hist, transition_matrix):
    fig = plt.figure(figsize=(18, 12))

    # Pixels
    ax1 = fig.add_subplot(131, projection='3d')
    x_hist = np.arange(len(hist))
    y_hist = np.zeros_like(hist)
    ax1.bar(x_hist, hist, zs=0, zdir='y', alpha=0.8, color='blue', width=0.5)
    ax1.set_title('Probabilities')
    ax1.set_xlabel('Color')
    # ax1.set_ylabel('Counts')
    ax1.set_zlabel('Probability')

    # Entropy
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.bar(['Entropy'], [entropy], zs=0, zdir='y', alpha=0.8, color='orange')
    ax2.set_title('Hartley Entropy')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Counts')
    ax2.set_zlabel('Entropy Value')

    # Markov Process
    ax3 = fig.add_subplot(133, projection='3d')
    x = np.arange(transition_matrix.shape[0])
    y = np.arange(transition_matrix.shape[1])
    x, y = np.meshgrid(x, y)

    z = transition_matrix.flatten()

    ax3.bar3d(x.flatten(), y.flatten(), np.zeros_like(z), 1, 1, z, shade=True)
    ax3.set_title('Transition Matrix')
    ax3.set_xlabel('Current Pixel Intensity')
    ax3.set_ylabel('Next Pixel Intensity')
    ax3.set_zlabel('Probability')

    plt.tight_layout()
    plt.show()


image = cv2.imread('images/I23.BMP')

height, width, channels = image.shape

probs = get_probs(image, width, height)
entropy = get_entropy(probs)
hist = get_histogram(image)
transition_matrix = markov_process(image)
plot_combined_3d(probs, transition_matrix)
