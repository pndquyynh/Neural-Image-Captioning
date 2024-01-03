import cv2
import numpy as np

def bgremove(myimage):
    # Create a copy of the input image to avoid modifying the original
    image = myimage.copy()

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Invert the image so the text is white and the background is black
    inverted_image = cv2.bitwise_not(blur)

    # Apply adaptive thresholding
    _, threshold_image = cv2.threshold(inverted_image, 128, 255, cv2.THRESH_BINARY)

    # Use adaptive thresholding to create a binary mask
    mask = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=4)

    # Make the entire background black
    image[mask == 0] = [0, 0, 0]

    # Make the text white
    image[mask == 255] = [255, 255, 255]

    kernel_erode = np.ones((1, 1), np.uint8)
    erosion_image = cv2.erode(image, kernel_erode, iterations=1)


    kernel_dialiation = np.ones((5, 5), np.uint8)

    diliation_image = cv2.morphologyEx(erosion_image, cv2.MORPH_OPEN, kernel_dialiation, iterations=1)

    # kernel_opening = np.ones((1, 1), np.uint8)
    #
    # opening_image = cv2.morphologyEx(diliation_image, cv2.MORPH_OPEN, kernel)

    # Return the final processed image
    return diliation_image

# Read the image
myimage = cv2.imread('F:\\Current_Topics\\Code\\Test_2.png')

# Now you can use myimage in your function
finalimage = bgremove(myimage.copy())

# Display the original and processed images
cv2.imshow('Original Image', myimage)
cv2.imshow('Processed Image', finalimage)
cv2.waitKey(0)
cv2.destroyAllWindows()