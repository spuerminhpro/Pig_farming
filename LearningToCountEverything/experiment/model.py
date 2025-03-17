import torch
import torch.nn as nn
import torch, torchvision
import torch.nn.functional as F
from collections import OrderedDict

class Resnet50FPN(nn.Module):
    def __init__(self):
        super(Resnet50FPN, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        children = list(self.resnet.children())
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]
    def forward(self, im_data):
        feat = OrderedDict()
        feat_map = self.conv1(im_data)
        feat_map = self.conv2(feat_map)
        feat_map3 = self.conv3(feat_map)
        feat_map4 = self.conv4(feat_map3)
        feat['map3'] = feat_map3
        feat['map4'] = feat_map4
        return feat

class DirectCountingModel(nn.Module):
    def __init__(self):
        super(DirectCountingModel, self).__init__()
        self.feature_extraction = Resnet50FPN()
        self.regressor = nn.Sequential(
            nn.Conv2d(1536, 196, kernel_size=7, padding=3),  # Match input channels (1024 + 512)
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU()  # Ensure non-negative density
        )

    def forward(self, query_image):
        image_features = self.feature_extraction(query_image)
        map3 = image_features['map3']  # [B, 1024, H/8, W/8]
        map4_upsampled = F.interpolate(image_features['map4'], scale_factor=2, mode='bilinear', align_corners=False)  # [B, 512, H/8, W/8]
        combined_feat = torch.cat([map3, map4_upsampled], dim=1)  # [B, 1536, H/8, W/8]
        density_map = self.regressor(combined_feat)  # [B, 1, H, W]
        return density_map