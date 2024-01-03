import cv2
import numpy as np

def bgremove(myimage):
    # Create a copy of the input image to avoid modifying the original
    image = myimage.copy()

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_image, 50, 150)

    # Apply morphological operations to enhance edges and remove small noise
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a binary mask to focus on text regions
    mask = np.zeros_like(edges)

    # Filter contours based on area (adjust the threshold as needed)
    min_contour_area = 200
    for contour in contours:
        if cv2.contourArea(contour) > min_contour_area:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Make the entire background black
    image[mask == 0] = [0, 0, 0]

    # Make the text white
    image[mask == 255] = [255, 255, 255]

    # Display the original, processed, and edge-detected images
    cv2.imshow('Original Image', myimage)
    cv2.imshow('Processed Image', image)
    cv2.imshow('Edges', edges)
    cv2.imshow('Mask', mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Read the image
myimage = cv2.imread('F:\\Current_Topics\\Code\\Test_2.png')

# Now you can use myimage in your function
bgremove(myimage.copy())
