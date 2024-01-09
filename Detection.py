import pickle

import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
import cv2 as cv
from craft_text_detector import Craft
from Classifier import Net, ResidualBlock
from torchvision import transforms

def bgremove(myimage):
    # Create a copy of the input image to avoid modifying the original
    image = myimage.copy()
    #gray_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    #gray_image = cv2.bitwise_not(image)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

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

    #kernel_erode = np.ones((1, 1), np.uint8)
    #erosion_image = cv2.erode(image, kernel_erode, iterations=1)


    #kernel_dialiation = np.ones((3, 3), np.uint8)
    #diliation_image = cv2.morphologyEx(erosion_image, cv2.MORPH_OPEN, kernel_dialiation, iterations=1)

    kernel_opening = np.ones((3, 3), np.uint8)
    opening_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_opening, iterations=1)

    # Return the final processed image
    return opening_image

    #return gray_image

def get_label_from_connected_components(box, label_ids_map):

    #print(box[0],box[1],box[2],box[3])

    return label_ids_map[int((box[1] + box[3])/2)][int((box[0] + box[2]) / 2)]

def generate_boxes(image):

    image_denoised = bgremove(image)
    craft = Craft(output_dir="./output", crop_type="box", cuda=False)
    prediction_result = craft.detect_text(image)
    #print("text_score", len(prediction_result["text_score_vector"]))
    text_score = prediction_result['text_score_vector']
    sure_bg = np.empty(shape=(len(text_score),len(text_score[0])),dtype=np.uint8)
    sure_bg.fill(255)
    for i in range(len(text_score)): #row
        for j in range(len(text_score[0])): #col
            if text_score[i][j] < 0.25: sure_bg[i][j] = 0

    sure_fg = np.empty(shape=(len(text_score),len(text_score[0])),dtype=np.uint8)
    sure_fg.fill(255)
    for i in range(len(text_score)): #row
        for j in range(len(text_score[0])): #col
            if text_score[i][j] < 0.35: sure_fg[i][j] = 0

    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)


    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    #print(image.shape)
    image = cv2.resize(image,(sure_fg.shape[1],sure_fg.shape[0]),interpolation=cv2.INTER_AREA)
    #print(image.shape)
    markers = cv.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    # Assuming 'markers' is your labeled input_image from cv2.connectedComponents()
    plt.imshow(markers)
    plt.show()

    unique_markers = np.unique(markers)  # Exclude 0 (background)

    # print(unique_markers)

    # print(markers)

    result = []

    for i, marker in enumerate(unique_markers):
        # if i == 0 or i == 1: continue
        # Get the coordinates of all pixels in this marker
        coords = np.where(markers == marker)

        # Get the bounding box coordinates
        x_min, y_min = np.min(coords[1]), np.min(coords[0])
        if (x_min < 5): continue
        x_max, y_max = np.max(coords[1]), np.max(coords[0])

        result.append((int(x_min * 98.5 / 100),
                       int(y_min * 98.5 / 100),
                       int(x_max * 101.5 / 100),
                       int(y_max * 101.5 / 100)))

        # Now (x_min, y_min) is the top-left coordinate of the bounding box, and (x_max, y_max) is the bottom-right.
        # You can use these to draw the bounding box on your input_image.
        # cv2.rectangle(input_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green box with thickness 2

    #for box in result:
    #    print(box[0], box[1], box[2], box[3])
    #    if box[2] == 117:
    #        cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0))

    #cv2.imshow('image',image)
    #cv2.waitKey(0)

    link_score = prediction_result['link_score_vector']
    link_score = (np.clip(link_score, 0, 1) * 255).astype(np.uint8)
    _,link_score = cv2.threshold(link_score, 128, 255, cv2.THRESH_BINARY)

    #cv2.imshow("link",link_score)
    #cv2.waitKey(0)
    totalLabels, label_ids = cv2.connectedComponents(link_score)
    print(label_ids.shape)
    #print("no_of_res",len(result))
    #print("total labels: ", totalLabels)
    #print("label_ids: ", label_ids)


    result.sort(key=lambda box: [get_label_from_connected_components(box,label_ids),box[0]])

    image_denoised = cv2.resize(image_denoised,(sure_bg.shape[1],sure_bg.shape[0]),interpolation=cv2.INTER_CUBIC)
    #image_denoised[sure_bg == 0] = (0, 0, 0)
    #cv2.imshow('image_denoised',image_denoised)
    #cv2.waitKey(0)


    sentences = []
    sentence = []

    sentence_level_coordinates = []
    letter_level_coordinates = []

    prev_end_of_sentence = False

    for i, box in enumerate(result):
        x_min, y_min, x_max, y_max = box

        # Crop and resize the image
        cropped_img = image_denoised[y_min:y_max, x_min:x_max]
        resized_img = cv2.resize(cropped_img, (50,50), interpolation=cv2.INTER_AREA)
        resized_img = cv2.copyMakeBorder(resized_img, 14, 14, 14, 14, cv2.BORDER_CONSTANT,
                                         None, value = 0)

        # Read pixel values and channels from the resized image
        # pixels = resized_img
        cv2.imshow ("abc", resized_img)
        cv2.waitKey(delay=0)

        if prev_end_of_sentence == True:
            sentences.append(sentence)
            sentence_level_coordinates.append(letter_level_coordinates)
            sentence = []
            letter_level_coordinates = []

        sentence.append(resized_img)
        letter_level_coordinates.append(box)
        # Add the bounding box coordinates to the separate list
        letter_level_coordinates.append((x_min, y_min, x_max, y_max))

        if i < len(result)-1:
            if get_label_from_connected_components(box,label_ids) != get_label_from_connected_components(result[i+1],label_ids):
                prev_end_of_sentence = True
                print(i, prev_end_of_sentence)
            else: prev_end_of_sentence = False
        else:
            sentences.append(sentence)


    #cv.imshow('image', image)
    #cv.imshow("Background_mask", sure_bg)
    #cv.imshow("Score_mask",sure_fg)
    #cv.waitKey(0)

    return result, sure_bg, sentences, sentence_level_coordinates

def classify(sentence_pics, sentence_coord ,model_path):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),  # Resize the image to the size expected by the model
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    m_location = torch.device('cpu') if torch.cuda.is_available() is False else torch.device('gpu')
    print(torch.cuda.is_available())

    model = torch.load("./best_cls_4096_6953_98_good.pt", map_location=m_location)
    print("Model loaded")

    # Check if a GPU is available and if not, use a CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.eval()
    for sentence in sentence_pics:
        for pic in sentence:
            pic = transform(pic).unsqueeze(0)

            # Move the model and image to the device
            model = model.to(device)
            image = pic.to(device)
            # Evaluation phase
            with torch.no_grad():
                # Forward pass
                outputs = model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                # something, predicted = torch.max(outputs.data, 1)
                max_prob, predicted = torch.max(probabilities, dim=1)
                print(max_prob.item() * 100, "%")
                with open('class_to_dir_4096_6953_98.pkl', 'rb') as f:
                    class_to_dir = pickle.load(f)
                    class_unicode = class_to_dir[predicted.item()]
                    print(f'Predicted class: {class_unicode}')
                    print(chr(int(class_unicode[2:], 16)))


ret, s_bg, sentences, sentence_level_coordinates = generate_boxes(cv.imread('./test3.png'))
classify(sentences,sentence_level_coordinates,"./best_cls_4096_6953_98_good.pt")



'''
image = cv2.imread('./test3.png')
ret,sure_bg = generate_boxes(image)

sure_bg = cv2.resize(sure_bg,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_CUBIC)
image_denoise = bgremove(image)
image_denoise[sure_bg==0] = (0,0,0)
for box in ret:
       cv.rectangle(image_denoise, (box[0]*2, box[1]*2), (box[2]*2, box[3]*2), (0, 255, 0))
cv.imshow('image_denoise',image_denoise)
cv.waitKey(0)
'''
