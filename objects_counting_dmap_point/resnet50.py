import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict

# Improved Loss: combine pixel-wise MSE loss with global count L1 loss.
class CombinedCountingLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        """
        Args:
            alpha: weight for the density map reconstruction loss (MSE).
            beta: weight for the global count loss (L1).
        """
        super(CombinedCountingLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred, gt):
        # Pixel-wise density map loss
        loss_density = self.mse(pred, gt)
        # Global counting loss: difference between the sums of the predicted and ground truth maps.
        count_pred = torch.sum(pred, dim=[1,2,3])
        count_gt = torch.sum(gt, dim=[1,2,3])
        loss_count = self.l1(count_pred, count_gt)
        return self.alpha * loss_density + self.beta * loss_count

# Using a pretrained ResNet50 as a backbone (Feature Pyramid Network)
import torch.nn as nn
import torchvision
from collections import OrderedDict

class Resnet50FPN(nn.Module):
    def __init__(self):
        super(Resnet50FPN, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        children = list(self.resnet.children())
        # children: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        self.conv1 = nn.Sequential(*children[:4])   # conv1, bn1, relu, maxpool
        self.layer1 = children[4]  # layer1 (256 channels)
        self.layer2 = children[5]  # layer2 (512 channels)
        self.layer3 = children[6]  # layer3 (1024 channels)
        self.layer4 = children[7]  # layer4 (2048 channels)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        feat_map3 = self.layer3(x)  # 1024 channels
        feat_map4 = self.layer4(feat_map3)  # 2048 channels
        feat = OrderedDict()
        feat['map3'] = feat_map3
        feat['map4'] = feat_map4
        return feat


# Improved Count Regressor using multi-scale features.
class CountRegressor(nn.Module):
    def __init__(self, input_channels, pool='mean'):
        super(CountRegressor, self).__init__()
        self.pool = pool
        # The regressor upsamples feature maps to reconstruct a full-resolution density map.
        # You can experiment with deeper or skip-connected architectures.
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, im):
        # Process each sample individually and then aggregate according to pooling mode.
        num_sample = im.shape[0]
        outputs = []
        for i in range(num_sample):
            out = self.regressor(im[i].unsqueeze(0))
            if self.pool == 'mean':
                out = torch.mean(out, dim=(2,3), keepdim=True)
            elif self.pool == 'max':
                out, _ = torch.max(out, dim=2, keepdim=True)
                out, _ = torch.max(out, dim=3, keepdim=True)
            outputs.append(out)
        return torch.cat(outputs, dim=0)

# Example combined model that uses the FPN and then the regressor.
class CountingModel(nn.Module):
    def __init__(self, pool='mean'):
        super(CountingModel, self).__init__()
        # Use the pretrained Resnet50FPN to extract multi-scale features.
        self.backbone = Resnet50FPN()
        # For instance, concatenate the feature maps from map3 and map4.
        # Here we assume they have the same spatial dimensions or are upsampled appropriately.
        self.count_regressor = CountRegressor(input_channels= (1024 + 2048), pool=pool)

    def forward(self, x):
        feats = self.backbone(x)
        # Upsample the lower-resolution map (map4) to match map3.
        map4_up = nn.functional.interpolate(feats['map4'], size=feats['map3'].shape[2:], mode='bilinear', align_corners=False)
        # Concatenate along the channel dimension.
        fused_feats = torch.cat([feats['map3'], map4_up], dim=1)
        # Pass through the regressor to obtain the density map.
        density_map = self.count_regressor(fused_feats)
        return density_map

# Example usage:
if __name__ == '__main__':
    model = CountingModel(pool='mean')
    # Dummy input: batch size of 2, RGB image 256x256.
    dummy_input = torch.randn(2, 3, 256, 256)
    output = model(dummy_input)
    print("Output shape:", output.shape)
