import pickle

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.backends import cudnn
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