import cv2
import numpy as np
import matplotlib.pyplot as plt



def plot_hists(segments, size):

    for i, segment in enumerate(segments):
        plt.figure()
        color = 'black'
        plt.hist(segment.ravel(), 256,[0,256], color = color)
        plt.title(f'Segment №{i+1} [{size}X{size}]')
        plt.xlabel('Intensity')
        plt.ylabel('N(Pixels)')
        plt.show()

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

image = cv2.imread('images/I23.BMP')


height, width, channels = image.shape

