import torch as tc
import torch.nn as nn
import torch.functional as F
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

save_path = 'SRGAN_RESULTS'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

latent_sz =100
batch_sz=128
epoch_sz=30
lr=0.0002
ims_sz=64

#하나의 블럭 통으로 구현.
class Block_B(nn.Module):
    def __init__(self, input_sz):
        super(Block_B, self).__init__()
        self.conv1 = nn.Conv2d(input_sz, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        step1= F.prelu(self.bn1(self.conv1(x)))
        return x + self.bn2(self.conv2(step1))

#TODO : 블럭 수 확인 필요.
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
#TODO : padding 4 맞는지 확인하기.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4)
#TODO : for문을 이용하여 쌓는 방법 고려하기.
        self.block1 = Block_B(64)
        self.block2 = Block_B(64)
        self.block3 = Block_B(64)
        self.block4 = Block_B(64)
        self.block5 = Block_B(64)
        self.block6 = Block_B(64)
        self.block7 = Block_B(64)
        self.block8 = Block_B(64)
        self.block9 = Block_B(64)
        self.block10 = Block_B(64)
        self.block11 = Block_B(64)
        self.block12 = Block_B(64)
        self.block13 = Block_B(64)
        self.block14 = Block_B(64)
        self.block15 = Block_B(64)
        self.block16 = Block_B(64)








#맞나 모르겠네...
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(512)

        self.flatten = nn.AdaptiveAvgPool2d(1)
        self.conv9 = nn.Conv2d(512, 1024, kernel_size=1)
        self.conv10 = nn.Conv2d(1024, 1, kernel_size=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn1(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv6(x)), 0.2)
        x = F.leaky_relu(self.bn6(self.conv7(x)), 0.2)
        x = F.leaky_relu(self.bn7(self.conv8(x)), 0.2)
        x = F.leaky_relu(self.conv9(self.flatten(x)))
        x = F.sigmoid(self.conv10(x))

        return x





