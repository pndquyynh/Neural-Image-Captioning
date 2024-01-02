import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont


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
        self.fc = nn.Linear(4096, 6953) #replace 2nd parameter with no. classes
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


# Define the transformations
transform = transforms.Compose([
    transforms.Grayscale(),  # Resize the image to the size expected by the model
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])



# Load the image
image = Image.open('./test.png')
'''
draw = ImageDraw.Draw(image)

# Specify the font, size, and color
font = ImageFont.truetype('./fonts/aoyagireisyosimo_ttf_2_01.ttf', 50)
text_color = (255, 255, 255)  # RGB for white

# Fill the image with black color
draw.rectangle([(0,0), image.size], fill=(0,0,0))
char_test = "動"
# Add text to the image
draw.text((32, 32), char_test, font=font, fill="white", anchor="mm")
print(char_test)

# Save the image
image.save('test.png')
'''

# Apply the transformations and add an extra dimension for the batch
image = transform(image).unsqueeze(0)

# Load the best model
model = torch.load("./best_cls_4096_6953_98_good.pt")
print("Model loaded")

# Check if a GPU is available and if not, use a CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Move the model and image to the device
model = model.to(device)
image = image.to(device)

# Evaluation phase
model.eval()
with torch.no_grad():
    # Forward pass
    outputs = model(image)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    #something, predicted = torch.max(outputs.data, 1)
    max_prob, predicted = torch.max(probabilities, dim=1)
    print(max_prob.item() * 100, "%")

# Load the mapping from a file
with open('class_to_dir_4096_6953_98.pkl', 'rb') as f:
    class_to_dir = pickle.load(f)
    class_unicode = class_to_dir[predicted.item()]
    print(f'Predicted class: {class_unicode}')
    print(chr(int(class_unicode[2:], 16)))
