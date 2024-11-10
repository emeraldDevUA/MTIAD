import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load a .bmp image
def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Unable to read the image.")
    return image

# Select a 1D segment (row or column) from the image
def get_image_segment(image, row=True, index=0):
    return image[index, :] if row else image[:, index]

# Perform linear regression on a segment
def linear_regression_analysis(segment):
    X = np.arange(len(segment)).reshape(-1, 1)
    y = segment

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return y_pred, mse

# Restore the image with linear regression applied on smaller sub-segments and calculate MSE
def restore_image_with_segments(image, segment_length=32, row=True):
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
            approximated_sub_segment = np.clip(approximated_sub_segment, 50, 205).astype(np.uint8)
            restored_segment.extend(approximated_sub_segment)

        # Convert restored segment to numpy array and place it in the image
        restored_segment = np.array(restored_segment, dtype=np.uint8)
        if row:
            restored_image[index, :] = restored_segment
        else:
            restored_image[:, index] = restored_segment

    # Calculate the average MSE for the entire image
    average_mse = total_mse / segment_count
    return restored_image, average_mse, mse_values

# Visualize original and approximated segments
def plot_segments(original, approximated, mse):
    plt.figure(figsize=(10, 5))
    plt.plot(original, label='Original Segment')
    plt.plot(approximated, label=f'Approximated Segment (MSE={mse:.2f})', linestyle='--')
    plt.legend()
    plt.title("Image Segment Approximation using Linear Regression")
    plt.xlabel("Position in Segment")
    plt.ylabel("Pixel Intensity")
    plt.show()

# Main execution
image_path = "images/F-16.bmp"  # Specify your .bmp file path here
image = load_image(image_path)  # Load the image
segment = get_image_segment(image, row=True, index=50)  # Get a row segment (e.g., row 50)

# Perform linear regression on the segment and calculate MSE
approximated_segment, mse = linear_regression_analysis(segment)

# Plot the original and approximated segment
# plot_segments(segment, approximated_segment, mse)

# Restore the image using linear regression on sub-segments of each row and get the overall MSE and MSE values for each segment
restored_image, average_mse, mse_values = restore_image_with_segments(image, segment_length=32, row=True)

# Display the average MSE for the restored image
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
