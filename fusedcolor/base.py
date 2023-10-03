import cv2
import numpy as np
import os
from PIL import Image


def make_stencils(input_image_path: str, output_directory: str):
    # Load the input image
    image = cv2.imread(input_image_path)

    if image is None:
        print("Could not open or read the image.")
    else:
        # Ensure the output directory exists
        os.makedirs(output_directory, exist_ok=True)

        # Define the RGB values for your paint colors
        # spray_can_colors = [
        #     (255, 0, 0),   # Red 1
        #     (0, 255, 0),   # Green 2
        #     (0, 0, 255),   # Blue 3
        #     (0, 0, 0),     # Black 4
        #     (255, 255, 255)  # White 5
        #     # Add more colors as needed
        # ]

        # Define the number of layers based on the number of spray can colors
        # num_layers = len(spray_can_colors)
        num_layers = 16

        # Calculate the opacity increment for each layer (25% opacity per layer)
        # opacity_increment = 0.25

        # reduce the number of colors in the image to the number of spray can colors
        reduced_image = reduce_colors(image, num_layers)
        cv2.imwrite(os.path.join(output_directory,
                    'reduced.png'), reduced_image)

        # Create stencils for each layer and spray can color
        unique_colors = np.unique(reduced_image.reshape(-1, 3), axis=0)

        # Save each color separately
        for i, color in enumerate(unique_colors):
            # Create a filename for the stencil
            stencil_filename = os.path.join(
                output_directory, f'stencil_{i + 1}.png')

            # Create a mask for pixels with the current color
            mask = cv2.inRange(reduced_image, color, color)

            # Apply the mask to the original image
            color_image = cv2.bitwise_and(
                reduced_image, reduced_image, mask=mask)

            # Save the color image
            cv2.imwrite(stencil_filename, color_image)

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
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

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
    target_color_hsv = cv2.cvtColor(
        np.uint8([[target_color]]), cv2.COLOR_BGR2HSV)[0][0]

    # Calculate Euclidean distance between each pixel and target color
    distances = np.sqrt(np.sum((hsv_image - target_color_hsv) ** 2, axis=2))

    # Create binary mask based on distance threshold
    mask = np.where(distances < threshold, 255, 0).astype(np.uint8)

    # Apply mask to original image
    return cv2.bitwise_and(image, image, mask=mask)


def save_cmyk_channels(input_image_path: str, output_directory: str):

    # Load the PNG image
    image = Image.open(input_image_path)

    # Convert the image to CMYK color space
    cmyk_image = image.convert('CMYK')

    # Split CMYK channels
    cyan, magenta, yellow, black = cmyk_image.split()

    # Save each channel separately
    cyan.save(os.path.join(output_directory, 'cyan.png'))
    magenta.save(os.path.join(output_directory, 'magenta.png'))
    yellow.save(os.path.join(output_directory, 'yellow.png'))
    black.save(os.path.join(output_directory, 'black.png'))

    print("CMYK channels saved.")

def something(x):
    pass

def showLines(input_image_path: str, output_directory: str):

    img = cv2.imread(input_image_path)
    img=cv2.resize(img,(600,400),cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_directory, 'gray.png'), gray)

    cv2.namedWindow('image')
    # create trackbars for color change
    cv2.createTrackbar('min','image',0,255,something)
    cv2.createTrackbar('max','image',0,255,something)
    # create switch for ON/OFF functionality
    switch = "0 : OFF \n 1 : ON"
    cv2.createTrackbar(switch, 'image',0,1,something)
    while(1):
        # get current positions of four trackbars
        g = cv2.getTrackbarPos ('min', 'image')
        b = cv2.getTrackbarPos ('max', 'image')
        s = cv2.getTrackbarPos (switch, 'image')
        if s == 0:
            img=img
            edges=img
        if s==1:
            edges = cv2.Canny(img,g,b)
        cv2.imshow ('image', edges)
        k = cv2.waitKey (1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(output_directory, 'edges.png'), edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                            minLineLength=100, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(os.path.join(output_directory, 'lines.png'), img)


