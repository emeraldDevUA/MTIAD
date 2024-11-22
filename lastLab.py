import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct


def segment_image_no_overlap(image, segment_size):
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


def apply_dct(segment):
    """
    Apply the Discrete Cosine Transform (DCT) to a 2D segment and round the coefficients to integers.

    Parameters:
    - segment: 2D NumPy array.

    Returns:
    - dct_segment: DCT-transformed 2D NumPy array with integer coefficients.
    """
    # Apply DCT
    dct_segment = dct(dct(segment.T, type=2, norm='ortho').T, type=2, norm='ortho')

    # Round coefficients to integers
    return np.round(dct_segment).astype(int)


def apply_idct(dct_segment):
    return idct(idct(dct_segment.T, type=2, norm='ortho').T, type=2, norm='ortho')



def round_coefficients(dct_image):

    return np.round(dct_image)


def calculate_rmse(original, reconstructed):
    """Calculate the Root Mean Square Error (RMSE) between two images."""
    diff = original - reconstructed
    mse = np.mean(diff ** 2)
    return np.sqrt(mse)


def process_image(original_image):
    """Main function to process the image."""
    # Read the image in grayscale

    if original_image is None:
        raise FileNotFoundError("Image not found!")

    # Apply DCT
    dct_image = apply_dct(original_image)

    # Round coefficients
    rounded_dct_image = round_coefficients(dct_image)

    # Reconstruct the image using inverse DCT
    reconstructed_image = apply_idct(rounded_dct_image)

    # Calculate RMSE
    rmse = calculate_rmse(original_image, reconstructed_image)

    # Plot the results
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("DCT Coefficients")
    plt.imshow(np.log1p(np.abs(dct_image)), cmap='gray')  # Log scale for visualization
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Reconstructed Image\\nRMSE: {rmse:.2f}")
    plt.imshow(reconstructed_image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return reconstructed_image, rmse

def apply_dct_to_rows(segment):
    dct_rows = []
    for row in segment:
        dct_row = cv2.dct(np.float32(row).reshape(-1, 1))  # Apply DCT to the row
        dct_rows.append(dct_row.flatten())  # Flatten back to 1D
    return np.array(dct_rows)

def calculate_mse(original_row, reconstructed_row):
    """
    Calculate the Mean Squared Error (MSE) between the original and reconstructed rows.

    Parameters:
    - original_row: 1D NumPy array representing the original row.
    - reconstructed_row: 1D NumPy array representing the reconstructed row.

    Returns:
    - mse: Mean Squared Error (MSE) as a float.
    """

    print(f"Original row shape: {original_row.shape}")
    print(f"Reconstructed row shape: {reconstructed_row.shape}")
    if original_row.shape != reconstructed_row.shape:
        raise ValueError("Original and reconstructed rows must have the same shape.")

    # Compute the squared differences
    squared_diff = (original_row - reconstructed_row) ** 2

    # Calculate and return the mean of squared differences
    mse = np.mean(squared_diff)
    return mse
def process_segment(segment):
    """
    Divide a segment into rows and plot their pixel values in a grid.
    Parameters:
    - segment: 2D NumPy array of shape (n, n).
    """
    rows, cols = segment.shape

    # Calculate the number of rows and columns for the grid
    num_rows = 4
    num_cols = int(np.ceil(rows / num_rows))

    # Create a single plot with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))  # Adjust figsize as needed

    for i, row in enumerate(segment):
        # Calculate the row and column index for the current subplot
        row_idx = i // num_cols
        col_idx = i % num_cols

        reconstructed_row = apply_dct_to_rows(row)
        # Plot pixel values for the current row
        axes[row_idx, col_idx].plot(row, label=f"Original Row {i} Values", marker='o')
        axes[row_idx, col_idx].plot(reconstructed_row, label=f"Reconstructed Row {i} Values", marker='x', color='orange')
        axes[row_idx, col_idx].set_title(f"Row {i}, MSE = {calculate_mse(row, reconstructed_row.flatten()):.4f}")
        axes[row_idx, col_idx].set_xlabel("Index")
        axes[row_idx, col_idx].set_ylabel("Value")

        # Add a legend
        axes[row_idx, col_idx].legend()

    # Adjust layout to avoid overlapping elements
    plt.tight_layout()
    plt.show()


def process_sq_segment(segment):
    """
    Process a square segment, apply DCT and inverse DCT, calculate MSE,
    and plot the original and reconstructed pixel values in a single graph.

    Parameters:
    - segment: 2D NumPy array of shape (n, n).
    """
    if segment.shape[0] != segment.shape[1]:
        raise ValueError("Segment must be square (n x n).")

    # Apply DCT to the entire segment
    dct_segment = apply_dct(segment)

    # Reconstruct the segment using inverse DCT
    reconstructed_segment = apply_idct(dct_segment)

    # Calculate MSE between the original and reconstructed segments
    mse = calculate_mse(segment.flatten(), reconstructed_segment.flatten())

    # Flatten the segments for plotting
    original_values = segment.flatten()
    reconstructed_values = reconstructed_segment.flatten()

    # Plot the original and reconstructed values
    plt.figure(figsize=(12, 6))
    plt.plot(original_values, label="Original Values", marker='o', linestyle='-', color='blue')
    plt.plot(reconstructed_values, label="Reconstructed Values", marker='x', linestyle='--', color='orange')
    plt.title(f"Original vs Reconstructed Segment (MSE: {mse:.4f})")
    plt.xlabel("Pixel Index")
    plt.ylabel("Pixel Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def fetch_segments(entropy_values, min_entropy, max_entropy):
    """
        Fetch the first segment in each category (low, medium, high) based on normalized entropy.

        Parameters:
        - entropy_values: List of entropy values to evaluate.
        - min_entropy: Minimum entropy value for normalization.
        - max_entropy: Maximum entropy value for normalization.

        Returns:
        - A dictionary with the first segment in each category: {'low': index, 'medium': index, 'high': index}.
          If a category is not found, its value will be None.
        """
    # Initialize placeholders for each category
    categories = {'low': None, 'medium': None, 'high': None}

    for i, entropy in enumerate(entropy_values):
        # Normalize the entropy
        normalized = (entropy - min_entropy) / (max_entropy - min_entropy)

        # Determine the category
        if normalized < 0.45 and categories['low'] is None:
            categories['low'] = i  # First low segment
        elif 0.45 <= normalized <= 0.55 and categories['medium'] is None:
            categories['medium'] = i  # First medium segment
        elif normalized > 0.55 and categories['high'] is None:
            categories['high'] = i  # First high segment

        # Break the loop early if all categories are found
        if all(value is not None for value in categories.values()):
            break

    return categories


def mean_arithmetical_expectation(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixel_values = image.flatten()
    sum_value = np.sum(pixel_values)

    return sum_value / len(pixel_values)


# Path to the image
image_path = "images/F-16.bmp"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

height, width = original_image.shape

segment_size = 8
segment_array = segment_image_no_overlap(original_image, segment_size)
brightness = []

for segment in segment_array:
    brightness.append(mean_arithmetical_expectation(segment))

critical_segments = fetch_segments(brightness, np.min(brightness), np.max(brightness))

process_segment(segment_array[critical_segments['low']])
process_segment(segment_array[critical_segments['medium']])
process_segment(segment_array[critical_segments['high']])

process_sq_segment(segment_array[critical_segments['low']])
process_sq_segment(segment_array[critical_segments['medium']])
process_sq_segment(segment_array[critical_segments['high']])

rmse = process_image(original_image)[1]

print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
