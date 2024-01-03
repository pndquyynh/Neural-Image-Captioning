import numpy as np
import cv2
from matplotlib import pyplot as plt
import cv2 as cv
from craft_text_detector import Craft

# Define global_image at the top of your script
remove = None

'''bgremove thay vì use biến global => trả return bth => gửi về đc global_image (đã đc remove bg)
=> new variable: global image remove => copy => get image has been already removed noise => paste next other function

detect boxes from heat map => use this variable to detect (global image remove)

surebg = bg_remove  = (54,463) <= shape'''

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
    image[mask == 0] = 0

    # Make the text white
    image[mask == 255] = 255

    kernel_erode = np.ones((1, 1), np.uint8)
    erosion_image = cv2.erode(image, kernel_erode, iterations=1)

    kernel_dialiation = np.ones((5, 5), np.uint8)

    diliation_image = cv2.morphologyEx(erosion_image, cv2.MORPH_OPEN, kernel_dialiation, iterations=1)

    # Update global_image with the final processed image
    global_image = diliation_image
    return diliation_image

def generate_boxes_from_image(image):
    craft = Craft(output_dir="./output", crop_type="box", cuda=False)
    prediction_result = craft.detect_text(image)
    text_boxes = prediction_result["boxes"]

    print(text_boxes)
    return

def generate_heat_map(image):
    # set image path and export folder directory
    #image = './test.png'  # can be filepath, PIL image or numpy array
    #output_dir = 'outputs/'

    # create a craft instance
    craft = Craft(output_dir="./output", crop_type="box", cuda=False)

    # apply craft text detection and export detected regions to output directory
    # prediction_result = craft.detect_text(image)
    # Use global_image as the input to craft.detect_text
    prediction_result = craft.detect_text(image)
    # cv2.imshow("heat",prediction_result["heatmaps"]["text_score_heatmap"])
    # cv2.waitKey(0)

    # unload models from ram/gpu
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()

    return prediction_result["heatmaps"]["text_score_heatmap"]


def detect_boxes_from_heat_map(input_image):
    global global_image_remove  # Use the global keyword to indicate you're using the global variable
    global resized_global_image_remove
    # Load the input_image
    #input_image = cv2.imread('heatmap.png')

    # Define range of blue color in BGR
    lower_blue = np.array([0,0,0])
    upper_blue_weak = np.array([255,200,140])
    #upper_blue_strong = np.array([255,255,200])
    kernel = np.ones((3,3), np.uint8)

    # Create a mask that only includes blue pixels within a certain range
    #mask_strong = cv2.inRange(input_image1, lower_blue, upper_blue_strong)
    mask_weak = cv2.inRange(input_image, lower_blue, upper_blue_weak)

    # Convert all pixels where the mask is not zero to white
    #input_image1[mask_strong!=0] = (0,0,0)
    #input_image1[mask_strong==0] = (255,255,255)

    input_image[mask_weak!=0] = (0,0,0)
    input_image[mask_weak==0] = (255,255,255)

    # sure background area
    # sure_bg = cv.dilate(input_image, kernel, iterations=1)
    # sure_bg = cv2.cvtColor(sure_bg, cv.COLOR_BGR2GRAY)
    sure_bg = cv2.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    cv.imshow("sure_bg",sure_bg)

    global_image_remove = cv2.cvtColor(global_image_remove, cv2.COLOR_BGR2GRAY)
    resized_global_image_remove = cv2.resize(global_image_remove, (width, height), interpolation=cv2.INTER_AREA)
    cv.imshow("imshow_remove", resized_global_image_remove)
    print("resized_global_image_remove.shape", resized_global_image_remove.shape)
    print("sure_bg.shape", sure_bg.shape)

    # Now use the resized image to index global_image
    resized_global_image_remove[sure_bg == 0] = 0
    cv.imshow("res_bs", resized_global_image_remove)
    print("curr_shape", resized_global_image_remove.shape)
    #int("max_bg",np.max(sure_bg))



    #bitwiseAnd = cv2.bitwise_and(input_image1, input_image)




    #eroded = cv2.erode(input_image,kernel)

    #closing = cv2.morphologyEx(input_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    #opening = cv.morphologyEx(input_image, cv2.MORPH_OPEN, kernel, iterations=7)


    #input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)

    dist_transform = cv2.distanceTransform(cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY),
                                           cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    ret2, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
    sure_fg = sure_fg.astype(np.uint8)
    cv2.imshow("sure_fg",sure_fg)

    #print("max_fg",np.max(sure_fg))
    #sure_fg = cv2.cvtColor(sure_fg,cv2.COLOR_BGR2GRAY)
    # Finding unknown region
    #sure_fg = np.uint8(sure_fg)

    #print("sure_bg:",sure_bg.dtype)
    #print("sure_fg:",sure_fg.dtype)

    #print("sure_bg:",sure_bg.shape)
    #print("sure_fg:",sure_fg.shape)

    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv.watershed(input_image,markers)
    input_image[markers == -1] = [255,0,0]
    #cv.imshow('input_image_with marker',input_image)

    #ret3, markers = cv2.connectedComponents(cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY))

    #labels = cv2.watershed(input_image,markers)
    #print(markers.shape)
    #annotated = color.lab2rgb(makers,0)

    #cv2.imshow('Colored Grains', annotated)

    plt.imshow(markers)
    plt.show()

    # Assuming 'markers' is your labeled input_image from cv2.connectedComponents()


    unique_markers = np.unique(markers)  # Exclude 0 (background)

    #print(unique_markers)

    #print(markers)

    result = []

    for i, marker in enumerate(unique_markers):
        #if i == 0 or i == 1: continue
        # Get the coordinates of all pixels in this marker
        coords = np.where(markers == marker)

        # Get the bounding box coordinates
        x_min, y_min = np.min(coords[1]), np.min(coords[0])
        if (x_min < 5): continue
        x_max, y_max = np.max(coords[1]), np.max(coords[0])

        result.append((int(x_min * 98/100),
                       int(y_min * 98/100),
                       int(x_max * 100/100),
                       int(y_max * 100/100)))

        # Now (x_min, y_min) is the top-left coordinate of the bounding box, and (x_max, y_max) is the bottom-right.
        # You can use these to draw the bounding box on your input_image.
        #cv2.rectangle(input_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box with thickness 2




    #cv.imshow('eroded',eroded)
    #cv.imshow("closed",closing)
    #cv2.imshow("opening",opening)
    # cv.imshow('input_image1',input_image1)
    #cv.imshow('input_image',input_image)
    cv.imshow("sure_bg",sure_bg)
    cv.imshow("sure_fg",sure_fg)
    #cv.imshow("unknown",unknown)
    #cv.imshow('And',bitwiseAnd)
    #cv2.waitKey(0)


    return result

img = cv.imread('test2.png')

# Use bgremove to process the image and update global_image
global_image_remove = bgremove(img)

#generate_boxes_from_image(img)


cv.imshow('img',img)

# heat_map = generate_heat_map("test2.png")
# Use global_image as the input to generate_heat_map
heat_map = generate_heat_map(img)
print("img.shape", img.shape)

print("heat_map.shape", heat_map.shape)

height, width = heat_map.shape[:2]
print("height, width of heat_map: ",height, width)
resized_global_image_remove = cv2.resize(global_image_remove, (width, height), interpolation = cv2.INTER_AREA)
resized_img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)


cv.imshow('heat_map',heat_map)

boxes = detect_boxes_from_heat_map(heat_map)
#print(boxes)


# for box in boxes:
#     cv.rectangle(resized_global_image_remove,(box[0],box[1]),(box[2],box[3]),(0,0,0))

cv.imshow('resized_img',resized_img)
cv.imshow('resized_global_image_remove',resized_global_image_remove)

cv.waitKey(0)