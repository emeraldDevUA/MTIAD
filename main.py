import cv2
import numpy as np
import matplotlib.pyplot as plt
import math as m


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
            if r > g + 50 and r > b + 50:
                red_cnt += 1.0
            if b > r + 50 and b > g + 50:
                blue_cnt += 1.0
            if g > r + 50 and g > b + 50:
                green_cnt += 1.0
            if r > 150 and g > 150 and b < 100:
                yellow_cnt += 1.0
            if r > 200 and g > 200 and b > 200:
                white_cnt += 0.1
            else: other_color += 1.0
    N = 3
    p1 = round(red_cnt / total_cnt, N)
    p2 = round(blue_cnt / total_cnt, N)
    p3 = round(green_cnt / total_cnt, N)
    p4 = round(yellow_cnt / total_cnt, N)
    p5 = round(white_cnt / total_cnt, N)
    p6 = round(other_color/total_cnt, N)

    return [p1, p2, p3, p4, p5, p6]


def get_entropy(probs):
    final_value = 0
    for p in probs:
        if p > 0:
            final_value += p * m.log(p, 2)
    return -final_value


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


def plot_combined_3d(hist, entropy, transition_matrix):
    fig = plt.figure(figsize=(18, 12))

    # Pixels
    ax1 = fig.add_subplot(131, projection='3d')
    x_hist = np.arange(len(hist))
    y_hist = np.zeros_like(hist)
    ax1.bar(x_hist, hist, zs=0, zdir='y', alpha=0.8, color='blue', width=0.5)
    ax1.set_title('Histogram of Pixel Intensities')
    ax1.set_xlabel('Intensity')
    ax1.set_ylabel('Counts')
    ax1.set_zlabel('Frequency')

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
    ax3.set_title('3D Visualization of Transition Matrix')
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

plot_combined_3d(hist, entropy, transition_matrix)
