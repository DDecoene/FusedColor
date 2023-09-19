import cv2
import numpy as np
import os


def make_stencils(input_image_path: str, output_directory: str):
    # Load the input image
    image = cv2.imread(input_image_path)

    if image is None:
        print("Could not open or read the image.")
    else:
        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Define the RGB values for your paint colors
        spray_can_colors = [
            (255, 0, 0),   # Red 1
            (0, 255, 0),   # Green 2
            (0, 0, 255),   # Blue 3
            (0, 0, 0),     # Black 4
            (255, 255, 255)  # White 5
            # Add more colors as needed
        ]

        # Define the number of layers based on the number of spray can colors
        num_layers = len(spray_can_colors)

        # Calculate the opacity increment for each layer (25% opacity per layer)
        opacity_increment = 0.25

        # reduce the number of colors in the image to the number of spray can colors
        image = reduce_colors(image,num_layers)
        cv2.imwrite(os.path.join(output_directory,'reduced.png'), image)
        
        # Create stencils for each layer and spray can color
        for i, color in enumerate(spray_can_colors):

            # Create a filename for the stencil
            stencil_filename = os.path.join(
                output_directory, f'stencil_{i + 1}.png')

            # Apply the spray can color to the stencil
            stencil_image = group_pixels_nearest_color(image,color,2)

            # Save the stencil as an image with the spray can color and opacity
            cv2.imwrite(stencil_filename, stencil_image)

            print(f"Stencil {i + 1} saved to {stencil_filename}")

        print("Stencil generation completed.")

def reduce_colors(image, k):
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)

    # Convert the pixels to float type
    pixels = np.float32(pixels)

    # Define criteria for k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Perform k-means clustering
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert the centers to unsigned 8-bit integer type
    centers = np.uint8(centers)

    # Replace each pixel value with its nearby center
    segmented_image = centers[labels.flatten()]

    # Reshape the segmented image back to its original shape
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image


def group_pixels_nearest_color(image, target_color, threshold):

    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Convert target color to HSV color space
    target_color_hsv = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_BGR2HSV)[0][0]

    # Calculate Euclidean distance between each pixel and target color
    distances = np.sqrt(np.sum((hsv_image - target_color_hsv) ** 2, axis=2))

    # Create binary mask based on distance threshold
    mask = np.where(distances < threshold, 255, 0).astype(np.uint8)

    # Apply mask to original image
    return cv2.bitwise_and(image, image, mask=mask)
