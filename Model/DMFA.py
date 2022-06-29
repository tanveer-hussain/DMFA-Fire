from torch import nn
import torchvision
import torch
import torch.nn.functional as F
from Multi_head import MHSA

class DFClassifier(nn.Module):
    def __init__(self):
        super(DFClassifier, self).__init__()
        self.original_model = torchvision.models.resnet152(pretrained=True)
        self.features1 = nn.Sequential(
            *list(self.original_model.children())[:-5]
        )

        self.features2 = nn.Sequential(
            *list(self.original_model.children())[:-4]
        )

        self.features3 = nn.Sequential(
            *list(self.original_model.children())[:-3]
        )

        self.features4 = nn.Sequential(
            *list(self.original_model.children())[:-2]
        )
        # original_model = torchvision.models.densenet121(pretrained=True, progress=True, memory_efficient=True)
        # self.features = nn.Sequential(
        #     *list(original_model.features.children())[:-5]
        # )
        CNN_channels = 128
        self.classification_head = nn.Sequential(
            nn.Conv2d(CNN_channels, CNN_channels, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(CNN_channels, CNN_channels , kernel_size=3, stride=2, padding=1),
            nn.Conv2d(CNN_channels, CNN_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(CNN_channels, CNN_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(CNN_channels * 2, CNN_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(12544, 2048),
            nn.Dropout(p=0.2),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.Dropout(p=0.2),
            nn.ReLU(True),
            nn.Linear(1024, 2)

        )

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        #
        self.conv1 = DoubleConv(256,64)
        self.conv2 = DoubleConv(512, 64)
        self.conv3 = DoubleConv(1024,64)

        # in_channels, in_resolution,out_channels,out_resolution,heads,initial
        self.mhsa1 = MHSA(64, width=56, height=56, heads=4)
        self.mhsa2 = MHSA(64, width=28, height=28, heads=4)
        self.mhsa3 = MHSA(64, width=14, height=14, heads=4)
        # self.aspp_mhsa1 = Pyramid_block(64, 56, 64, 56, 4, 1)
        # self.aspp_mhsa2 = Pyramid_block(64, 28, 64, 28, 4, 2)
        # self.aspp_mhsa3 = Pyramid_block(64, 14, 64, 14, 4, 3)

        self.conv128_64 = DoubleConv(128,64)
        self.conv256_64 = DoubleConv(256, 64)
        self.conv64_32 = DoubleConv(64, 32)
        self.conv_pred = DoubleConv(32,1)


    def forward(self, x):
        # f1 = self.features1(x)  # [1, 256, 56, 56]
        f2 = self.features2(x) # [1, 512, 28, 28]
        f3 = self.features3(x) # [1, 1024, 14, 14]

        # f1 = self.conv1(f1)
        f2 = self.conv2(f2)
        f3 = self.conv3(f3)

        # f1 = self.mhsa1(f1)
        f2 = self.mhsa2(f2)
        f3 = self.mhsa3(f3)

        f32 = torch.cat((self.upsample2(f3),f2),1)
        # f321 = torch.cat((self.upsample2(f32), f1), 1)


        class_output = self.classification_head(f32)

        return class_output

# model = DFClassifier()
# # print (model)
# x = torch.randn(4,3,224,224)
# y  = model(x)
# print (y.shape)

