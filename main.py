import numpy as np
import cv2
from matplotlib import pyplot as plt
import cv2 as cv
from craft_text_detector import Craft

'''
def generate_boxes_from_image(image):
    craft = Craft(output_dir="./output", crop_type="box", cuda=False)
    prediction_result = craft.detect_text(image)
    text_boxes = prediction_result["boxes"]

    print(text_boxes)
    return
'''

def generate_heat_map(image):
    # set image path and export folder directory
    #image = './test.png'  # can be filepath, PIL image or numpy array
    #output_dir = 'outputs/'

    # create a craft instance
    craft = Craft(output_dir="./output", crop_type="box", cuda=False)

    # apply craft text detection and export detected regions to output directory
    prediction_result = craft.detect_text(image)
    # cv2.imshow("heat",prediction_result["heatmaps"]["text_score_heatmap"])
    # cv2.waitKey(0)

    # unload models from ram/gpu
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()

    return prediction_result["heatmaps"]["text_score_heatmap"]


def detect_boxes_from_heat_map(input_image):
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
    sure_bg = cv.dilate(input_image,kernel,iterations=1)
    sure_bg = cv2.cvtColor(sure_bg,cv.COLOR_BGR2GRAY)
    cv.imshow("sure_bg",sure_bg)
    #print("max_bg",np.max(sure_bg))

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

    #plt.imshow(markers)
    #plt.show()

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

        result.append((int(x_min * 98.5/100),
                       int(y_min * 98.5/100),
                       int(x_max * 101.5/100),
                       int(y_max * 101.5/100)))

        # Now (x_min, y_min) is the top-left coordinate of the bounding box, and (x_max, y_max) is the bottom-right.
        # You can use these to draw the bounding box on your input_image.
        #cv2.rectangle(input_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box with thickness 2




    #cv.imshow('eroded',eroded)
    #cv.imshow("closed",closing)
    #cv2.imshow("opening",opening)
    # cv.imshow('input_image1',input_image1)
    #cv.imshow('input_image',input_image)
    #cv.imshow("sure_bg",sure_bg)
    cv.imshow("sure_fg",sure_fg)
    cv.imshow("unknown",unknown)
    #cv.imshow('And',bitwiseAnd)
    #cv2.waitKey(0)


    return result

img = cv.imread('test2.png')


#generate_boxes_from_image(img)

cv.imshow('img',img)

heat_map = generate_heat_map("test2.png")
print(img.shape)

print(heat_map.shape)

height, width = heat_map.shape[:2]
resized_img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)


cv.imshow('heat_map',heat_map)

boxes = detect_boxes_from_heat_map(heat_map)
#print(boxes)

'''
for box in boxes:
    cv.rectangle(resized_img,(box[0],box[1]),(box[2],box[3]),(0,0,0))
'''

cv.imshow('resized_img',resized_img)

cv.waitKey(0)