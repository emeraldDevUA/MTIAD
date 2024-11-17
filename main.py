import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from PIL import Image, ImageDraw, ImageFont


# Load a .bmp image
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unable to read the image.")
    return image


# Select a 1D segment (row or column) from the image
def get_image_segment(image, row=True, index=0):
    return image[index, :] if row else image[:, index]


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


def linear_regression_analysis(segment):
    X = np.arange(len(segment)).reshape(-1, 1)
    y = segment

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return y_pred, mse


# Restore the image with linear regression applied on smaller sub-segments and calculate MSE
def restore_image_with_segments(image, segment_length=8, row=True):
    restored_image = np.zeros_like(image, dtype=np.uint8)
    dim_size = image.shape[0] if row else image.shape[1]

    total_mse = 0
    segment_count = 0
    mse_values = []

    # Loop through each row or column
    for index in range(dim_size):
        # Get the row or column segment
        full_segment = get_image_segment(image, row=row, index=index)

        # Process in sub-segments
        restored_segment = []
        for start in range(0, len(full_segment), segment_length):
            end = min(start + segment_length, len(full_segment))
            sub_segment = full_segment[start:end]

            # Apply linear regression on the sub-segment and get MSE
            approximated_sub_segment, mse = linear_regression_analysis(sub_segment)
            mse_values.append(mse)  # Collect MSE for each sub-segment
            total_mse += mse
            segment_count += 1

            # Clip values to stay within valid pixel range (50-205) and add to restored segment
            approximated_sub_segment = approximated_sub_segment.astype(np.uint8)

            restored_segment.extend(approximated_sub_segment)

        # Convert restored segment to numpy array and place it in the image
        restored_segment = np.array(restored_segment, dtype=np.uint8)
        if row:
            restored_image[index, :] = restored_segment
        else:
            restored_image[:, index] = restored_segment

    # Calculate the average MSE for the entire image
    average_mse = np.sqrt(total_mse / segment_count)
    return restored_image, average_mse, np.sqrt(mse_values)


def process_segment(segment):
    """
    Divide a segment into rows, plot them and their regression models in a 4x4 grid.

    Parameters:
    - segment: 2D NumPy array of shape (n, n).

    Returns:
    - None (plots results).
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

        # Perform linear regression
        predictions, _ = linear_regression_analysis(row)

        # Plot actual values and regression predictions on the current subplot
        axes[row_idx, col_idx].plot(row, label=f"Row {i} Values", marker='o')
        axes[row_idx, col_idx].plot(predictions, label=f"Regression (Row {i})", linestyle="--")

        # Add labels and title to each subplot
        axes[row_idx, col_idx].set_title(f"Row {i}: Values vs Regression")
        axes[row_idx, col_idx].set_xlabel("Index")
        axes[row_idx, col_idx].set_ylabel("Value")

    # Adjust layout to avoid overlapping elements
    plt.tight_layout()
    plt.show()

def plot_segments(original, approximated, mse):
    plt.figure(figsize=(10, 5))
    plt.plot(original, label='Original Segment')
    plt.plot(approximated, label='Approximated Segment', color='red')
    plt.plot(mse, label=f'Approximated Segment (MSE={mse:.2f})', linestyle='--')
    plt.legend()
    plt.title("Image Segment Approximation using Linear Regression")
    plt.xlabel("Position in Segment")
    plt.ylabel("Pixel Intensity")
    plt.show()


def entropy_to_color(entropy, min_entropy, max_entropy, plots=False):
    # Normalize entropy between 0 and 1
    normalized = (entropy - min_entropy) / (max_entropy - min_entropy)

    # Threshold normalized value to 0, 0.5, or 1
    if normalized > 0.7:
        normalized = 1
        colormap = plt.cm.viridis
    elif normalized < 0.3:
        normalized = 1
        colormap = plt.cm.plasma
    else:  # 0.45 <= normalized <= 0.55
        normalized = 0.5
        colormap = plt.cm.viridis

        # Convert to color using a colormap
    # Use 'viridis' or other colormaps
    color = colormap(normalized)  # Returns a tuple (R, G, B, A)

    if plots:
        return color[:3]
    else:
        return tuple([int(255 * c) for c in color[:3]])


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


# Main execution
image_path = "images/F-16.bmp"  # Specify your .bmp file path here
image = load_image(image_path)  # Load the image
height, width = image.shape

segment_size = 8

segment_array = segment_image_no_overlap(image, segment_size)

brightness = []

for segment in segment_array:
    brightness.append(mean_arithmetical_expectation(segment))


reconstruct_image(brightness, segment_size, (width, height), "reconstructed image").show()
critical_segments = fetch_segments(brightness, np.min(brightness), np.max(brightness))

restored_image, average_mse, mse_values = restore_image_with_segments(image, segment_length=segment_size, row=True)

process_segment(segment_array[critical_segments['low']])
process_segment(segment_array[critical_segments['medium']])
process_segment(segment_array[critical_segments['high']])

print(f"Average MSE for the restored image: {average_mse:.2f}")

# Plot the original and restored images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title(f"Restored Image (Row-wise Linear Regression)\nAverage MSE: {average_mse:.2f}")
plt.imshow(restored_image, cmap='gray')
plt.show()

# Plot MSE values for each segment
plt.figure(figsize=(10, 5))
plt.plot(mse_values, label="MSE for Each Segment", color='blue')
plt.axhline(y=average_mse, color='red', linestyle='--', label=f"Average MSE ({average_mse:.2f})")
plt.title("MSE per Segment in Restored Image")
plt.xlabel("Segment Index")
plt.ylabel("MSE")

plt.legend()
plt.show()
