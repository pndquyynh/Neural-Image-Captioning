Requirement: 

export_extra_results : comment out to prevent saving if necessary

vgg16bn.py:

#from torchvision.models.vgg import model_urls

all = [ 'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19', ]

model_urls = { 'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth', 'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth', 'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth', 'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth', 'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth', 'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth', 'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth', 'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth', }

resize_aspect_ratio in predict.py affect the scaling