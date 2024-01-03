import numpy as np
import cv2
from matplotlib import pyplot as plt
import cv2 as cv
from craft_text_detector import Craft
import os
import pickle

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.backends import cudnn

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
    # upper_blue_weak = np.array([0, 0, 0])
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

    print("Shape of sure_bg:", sure_bg.shape)
    #print("max_bg",np.max(sure_bg))



    #bitwiseAnd = cv2.bitwise_and(input_image1, input_image)




    #eroded = cv2.erode(input_image,kernel)

    #closing = cv2.morphologyEx(input_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    #opening = cv.morphologyEx(input_image, cv2.MORPH_OPEN, kernel, iterations=7)

    #input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)

    dist_transform = cv2.distanceTransform(cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY),
                                           cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    ret2, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
    sure_fg = sure_fg.astype(np.uint8)
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

        result.append((int(x_min * 99/100),
                       int(y_min * 99/100),
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

def crop_image(image, boxes):
    boxes.sort(key=lambda box: box[0])
    cropped_images = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        cropped_img = image[y_min:y_max, x_min:x_max]
        cropped_images.append(cropped_img)
    return cropped_images

def save_cropped_images(cropped_images, output_dir='letter_crops', target_size=(64, 64)):
    os.makedirs(output_dir, exist_ok=True)
    for i, cropped_img in enumerate(cropped_images):
        # Resize the image to the target size
        resized_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_AREA)

        # Save the resized image
        output_path = os.path.join(output_dir, f'letter_{i + 1}.png')
        cv2.imwrite(output_path, resized_img)
        print(f'Saved {output_path}')

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

# Crop the image based on the detected boxes
cropped_images = crop_image(resized_img, boxes)

# Display or save the cropped images as needed
for i, cropped_img in enumerate(cropped_images):
    cv.imshow(f'Cropped Image {i}', cropped_img)

# Save cropped images to files
save_cropped_images(cropped_images, target_size=(64, 64))

cv.waitKey(0)
# cls
class ResidualBlock(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        #relu
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # relu
        self.conv3 = nn.Conv2d(out_channels, out_channels= out_channels * self.expansion,
                               kernel_size= 1, stride = 1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )


    def forward(self, x):
        residual = self.skip(x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x += residual
        x = F.relu(x)
        return x

class Net(nn.Module): #[3,4,6,3]
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1,64, kernel_size=7, stride=2, padding=3)
        #use 64 filters to create 64 out channels
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.curr_in_channels = 64

        # Define each layer of blocks
        #self.layer1 = self.make_layer(ResidualBlock, 64, 64, 3)
        #self.layer2 = self.make_layer(ResidualBlock, 256, 1024, 4, stride=2)
        #self.layer3 = self.make_layer(ResidualBlock, 1024, 4096, 6, stride=2)
        #self.layer4 = self.make_layer(ResidualBlock, 4096, 16384, 3, stride=2)

        self.layer1 = ResidualBlock(64,64, 1)
        #output no. channel = 256
        self.layer2 = ResidualBlock(256, 64, 1)
        #self.layer3 = ResidualBlock(256,64, 1)

        #no. in_channel = 256
        self.layer4 = ResidualBlock(256,256,2)
        # output no. channel = 1024
        self.layer5 = ResidualBlock(1024,256,1)
        #self.layer6 = ResidualBlock(1024,256,1)
        #self.layer7 = ResidualBlock(1024,256,1)

        self.layer8 = ResidualBlock(1024, 1024 , 2)
        self.layer9 = ResidualBlock(4096, 1024, 1)
        #self.layer10 = ResidualBlock(4096, 1024, 1)
        #self.layer11 = ResidualBlock(4096, 1024, 1)
        #self.layer12 = ResidualBlock(4096, 1024, 1)
        #self.layer13 = ResidualBlock(4096,1024,1)

        #self.layer14 = ResidualBlock(4096, 4096,2)
        #self.layer15 = ResidualBlock(16384, 4096, 1)
        #self.layer16 = ResidualBlock(16384, 4096, 1)
        #no .out_channel = 16384

        # Define the final fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4096, 6953)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.max_pool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        #x = self.layer6(x)
        #x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        #x = self.layer10(x)
        #x = self.layer11(x)
        #x = self.layer12(x)
        #x = self.layer13(x)
        #x = self.layer14(x)
        #x = self.layer15(x)
        #x = self.layer16(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

net = Net()
print(net)

# Define the transformations
transform = transforms.Compose([
    transforms.Grayscale()
    ,transforms.ToTensor()
    ,transforms.Normalize((0.5,), (0.5,))
])

# Load the datasets
train_data = datasets.ImageFolder('./dataset/train/', transform=transform)
val_data = datasets.ImageFolder('./dataset/valid/', transform=transform)

# Create data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
print("train data loaded : " + str(len(train_data)))

val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
print("val data loaded : " + str(len(val_data)))



# Create a mapping of class numbers to class names (i.e., subdirectory names)
class_to_dir = {i: c for c, i in train_data.class_to_idx.items()}

# Save the mapping to a file
with open('class_to_dir.pkl', 'wb') as f:
    pickle.dump(class_to_dir, f)

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = torch.cuda.is_available()
cudnn.benchmark = True
print(device)

# Move the model to the device
net = Net().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

prev_best_acc = 0

# Training the network
for epoch in range(100):  # loop over the dataset multiple times

    net.train()
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1} Training'):

        # Move the images and labels to the device
        images = images.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Testing the network on the test data
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1} Validation'):
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Validation Accuracy: {accuracy * 100}%')
    if accuracy >= prev_best_acc:
        torch.save(net, "./best.pt")
        prev_best_acc = accuracy
        print("Leading model checkpoint")

print('Training complete.')