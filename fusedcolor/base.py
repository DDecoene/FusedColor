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
            (255, 0, 0),   # Red
            (0, 255, 0),   # Green
            (0, 0, 255),   # Blue
            (0, 0, 0),     # Black
            (255, 255, 255)  # White
            # Add more colors as needed
        ]

        # Define the number of layers based on the number of spray can colors
        num_layers = len(spray_can_colors)

        # Calculate the opacity increment for each layer (25% opacity per layer)
        opacity_increment = 0.25

        # Create stencils for each layer and spray can color
        for i, color in enumerate(spray_can_colors):
            # Create a mask for the current layer with the desired opacity
            opacity = (i + 1) * opacity_increment
            layer_mask = np.ones_like(image, dtype=np.float32) * opacity

            # Create a filename for the stencil
            stencil_filename = os.path.join(
                output_directory, f'stencil_{i + 1}.png')

            # Apply the spray can color to the stencil
            stencil_image = np.ones_like(
                image, dtype=np.uint8) * np.array(color)
            painted_image = (stencil_image * layer_mask).astype(np.uint8)

            # Save the stencil as an image with the spray can color and opacity
            cv2.imwrite(stencil_filename, painted_image)

            print(f"Stencil {i + 1} saved to {stencil_filename}")

        print("Stencil generation completed.")
