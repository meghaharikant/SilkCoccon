import cv2
import numpy as np
import pandas as pd

# List of image paths
image_paths = [
    "C://Users//megha//OneDrive//ESM//silk_cocoon//0-2_png.rf.82f10fabdd399deecf8a3ac59ba456ea.jpg",
    "C://Users//megha//OneDrive//ESM//silk_cocoon//0-6_png.rf.871fc855371175bc25a49bb4e104ab1c.jpg",
    "C://Users//megha//OneDrive//ESM//silk_cocoon//1_4_png.rf.06f2ba676f2714f9572cf35530790a32.jpg",
    "C://Users//megha//OneDrive//ESM//silk_cocoon//3_3_png.rf.dc5a71605e5b82a038ea23f7573144e2.jpg",
    "C://Users//megha//OneDrive//ESM//silk_cocoon//3_4_png.rf.752f05f8496eec756448ea7050b9aa8b.jpg"
]

# Load the dataset of color information
color_dataset_path = "C://Users//megha//OneDrive//ESM//silk_cocoon//cocoon_colur.cocoon.csv"
color_df = pd.read_csv(color_dataset_path)

# Function to extract color features from the dataset based on image path
def get_color_features(image_path, color_df):
    # Find corresponding row in the color dataset based on image path
    row = color_df[color_df['Image_Path'] == image_path]
    if row.empty:
        print(f"Error: No color information found for image '{image_path}'")
        return None
    # Extract color features
    color_features = row.iloc[:, 1:].values.flatten()  # Assuming color features start from the second column
    return color_features

# Process each image
for image_path in image_paths:
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to read image '{image_path}'")
        continue

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Threshold the image to create a binary image
    _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Set a minimum threshold area to filter out small contours
        min_area = 100
        
        if area > min_area:
            # Draw the contour on the original image
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            
            # Get color features for the current image
            color_features = get_color_features(image_path, color_df)
            
    # Display the result
    cv2.imshow('Cocoon Detection - ' + image_path, image)

# Wait for key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
