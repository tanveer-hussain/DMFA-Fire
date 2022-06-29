from torch import nn
import torchvision
import torch
import torch.nn.functional as F
from Multi_head import MHSA


class Pyramid_block(nn.Module):
    def __init__(self, in_channels, in_resolution,out_channels,out_resolution,heads,initial):
        super(Pyramid_block, self).__init__()


        self.block1 = nn.ModuleList()

        if in_channels != out_channels:
            self.block1.append(DoubleConv(in_channels, out_channels))


        if initial==1:
            self.block1.append(multi_scale_aspp(in_channels))
            self.block1.append(multi_scale_aspp(in_channels))
            self.block1.append(multi_scale_aspp(in_channels))
            self.block1.append(MHSA(out_channels, width=in_resolution, height=in_resolution, heads=heads))
        elif initial==2:
            self.block1.append(multi_scale_aspp(in_channels))
            self.block1.append(multi_scale_aspp(in_channels))
            self.block1.append(MHSA(in_channels, width=in_resolution, height=in_resolution, heads=heads))
        elif initial==3:
            self.block1.append(multi_scale_aspp(in_channels))
            self.block1.append(MHSA(in_channels, width=in_resolution, height=in_resolution, heads=heads))
        elif initial==4:
            self.block1.append(multi_scale_aspp(in_channels))
        self.block1 = nn.Sequential(*self.block1)


        self.in_resolution = in_resolution
        self.out_resolution = out_resolution

    def forward(self, x):
        x = self.block1(x)
        if self.in_resolution != self.out_resolution:
            x = F.interpolate(x, size=(self.out_resolution,self.out_resolution), mode='bilinear',align_corners=True)

        return x


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

#
# class DFModel(nn.Module):
#     def __init__(self):
#         super(DFModel, self).__init__()
#         self.original_model = torchvision.models.resnet152(pretrained=True)
#         self.features1 = nn.Sequential(
#             *list(self.original_model.children())[:-5]
#         )
#
#         self.features2 = nn.Sequential(
#             *list(self.original_model.children())[:-4]
#         )
#
#         self.features2 = nn.Sequential(
#             *list(self.original_model.children())[:-4]
#         )
#
#         self.features3 = nn.Sequential(
#             *list(self.original_model.children())[:-3]
#         )
#
#         self.features4 = nn.Sequential(
#             *list(self.original_model.children())[:-2]
#         )
#         # original_model = torchvision.models.densenet121(pretrained=True, progress=True, memory_efficient=True)
#         # self.features = nn.Sequential(
#         #     *list(original_model.features.children())[:-5]
#         # )
#         CNN_channels = 256
#         self.classification_head = nn.Sequential(
#             # nn.Conv2d(CNN_channels, CNN_channels // 2, kernel_size=3, stride=1, padding=1),
#             # nn.Conv2d(CNN_channels // 2, CNN_channels // 2, kernel_size=3, stride=2, padding=1),
#             # nn.Conv2d(CNN_channels // 2, CNN_channels // 4, kernel_size=3, stride=1, padding=1),
#             # nn.ReLU(True),
#             # nn.Conv2d(CNN_channels // 4, CNN_channels // 4, kernel_size=3, stride=2, padding=1),
#             # nn.Conv2d(CNN_channels // 4, CNN_channels // 4, kernel_size=3, stride=1, padding=1),
#             # nn.ReLU(True),
#             nn.Flatten(),
#             nn.Linear(12544, 2048),
#             nn.ReLU(True),
#             nn.Linear(2048, 1024),
#             nn.ReLU(True),
#             nn.Linear(1024, 2)
#
#         )
#
#         self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
#
#         self.conv1 = DoubleConv(256,64)
#         self.conv2 = DoubleConv(512, 64)
#         self.conv3 = DoubleConv(1024,64)
#
#         # in_channels, in_resolution,out_channels,out_resolution,heads,initial
#         # self.aspp_mhsa1 = Pyramid_block(64, 56, 64, 56, 4, 1)
#         # self.aspp_mhsa2 = Pyramid_block(64, 28, 64, 28, 4, 2)
#         # self.aspp_mhsa3 = Pyramid_block(64, 14, 64, 14, 4, 3)
#
#         self.conv128_64 = DoubleConv(128,64)
#         self.conv256_64 = DoubleConv(256, 64)
#         self.conv64_32 = DoubleConv(64, 32)
#         self.conv_pred = DoubleConv(32,1)
#
#
#     def forward(self, x):
#         f1 = self.features1(x)  # [1, 256, 56, 56]
#         f2 = self.features2(x) # [1, 512, 28, 28]
#         f3 = self.features3(x) # [1, 1024, 14, 14]
#
#         f1 = self.conv1(f1)
#         f2 = self.conv2(f2)
#         f3 = self.conv3(f3)
#
#         # f1 = self.aspp_mhsa1(f1)
#         # f2 = self.aspp_mhsa2(f2)
#         # f3 = self.aspp_mhsa3(f3)
#
#         class_output = self.classification_head(f3)
#         # print (f1.shape, f2.shape, f3.shape, class_output.shape)
#
#         f32 = self.conv128_64(torch.cat((self.upsample2(f3),f2),1))
#         f321 = self.conv256_64(torch.cat((self.upsample2(f32), f1, self.upsample4(f3), self.upsample2(f2)), 1))
#         # print (f32.shape, f321.shape)
#
#         f321 = self.conv_pred(self.conv64_32(self.upsample4(f321)))
#
#         return class_output, f321
