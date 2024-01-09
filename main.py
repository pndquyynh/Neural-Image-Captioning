import numpy as np
import cv2
from matplotlib import pyplot as plt
import cv2 as cv
from craft_text_detector import Craft
import os

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
    sure_bg = cv.dilate(input_image,kernel,iterations=3)
    sure_bg = cv2.cvtColor(sure_bg,cv.COLOR_BGR2GRAY)
    cv.imshow("sure_bg", sure_bg)
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
    input_image[0] = [0,0,0]
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
    #cv.imshow("sure_bg",sure_bg)
    cv.imshow("sure_fg",sure_fg)
    #cv.imshow("unknown",unknown)
    #cv.imshow('And',bitwiseAnd)
    #cv2.waitKey(0)


    return result

# def crop_and_save_sentences(image, boxes, output_dir='letter_crops', target_size=(64, 64)):
#     os.makedirs(output_dir, exist_ok=True)
#     sentence_count = 1
#     sentence_dir = os.path.join(output_dir, f'sentence_{sentence_count}')
#
#     # # Sort boxes by x-coordinate
#     boxes.sort(key=lambda box: box[0])
#
#     cropped_data = []
#
#     for i, box in enumerate(boxes):
#         x_min, y_min, x_max, y_max = box
#
#         # Check if the distance between bounding boxes is large enough to start a new sentence
#         if i > 0 and x_min - boxes[i - 1][2] > 20:
#             sentence_count += 1
#             sentence_dir = os.path.join(output_dir, f'sentence_{sentence_count}')
#             os.makedirs(sentence_dir, exist_ok=True)
#
#         # Crop and resize the image
#         cropped_img = image[y_min:y_max, x_min:x_max]
#         resized_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)
#
#         # Save the resized image
#         output_path = os.path.join(sentence_dir, f'letter_{i + 1}.png')
#         cv2.imwrite(output_path, resized_img)
#         print(f'Saved {output_path}')
#
#         # Add information to the cropped_data
#         cropped_data.append([sentence_count, i + 1, cv2.mean(resized_img)])
#
#     return np.array(cropped_data, dtype=object)

def convert_to_3d_array(image, boxes, target_size=(64, 64)):
    # Sort boxes by x-coordinate
    boxes.sort(key=lambda box: box[0])

    sentences = []
    letter_coordinates = []

    sentence_count = 1
    sentence = []

    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box

        # Check if the distance between bounding boxes is large enough to start a new sentence
        if i > 0 and x_min - boxes[i - 1][2] > 20:
            sentences.append(sentence)
            sentence_count += 1
            sentence = []

        # Crop and resize the image
        cropped_img = image[y_min:y_max, x_min:x_max]
        resized_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)

        # Read pixel values and channels from the resized image
        pixels = resized_img.shape
        # cv2.imshow ("abc", resized_img)
        # cv2.waitKey(delay=0)

        # Add the pixel information to the letter list
        sentence.append(pixels)

        # # Add the resized image to the letter list
        # letter_info = cv2.imread(resized_img)
        # sentence.append(letter_info)
        # Add the bounding box coordinates to the separate list
        letter_coordinates.append((x_min, y_min, x_max, y_max))
    # Add the last sentence to the sentences list
    sentences.append(sentence)

    # Convert the list structure to a 3D array
    array_3d = np.array(sentences, dtype=object)

    return array_3d, letter_coordinates

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


for box in boxes:
    cv.rectangle(resized_img,(box[0],box[1]),(box[2],box[3]),(0,0,0))


cv.imshow('resized_img',resized_img)

# Convert to 3D array
array_3d = convert_to_3d_array(resized_img, boxes)

# Display the 3D array
print(array_3d)


cv.waitKey(0)

