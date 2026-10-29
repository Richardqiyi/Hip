import torch
from torchvision import models
import torch.nn as nn

class ResNet50FeatureExtractor3D(torch.nn.Module):
    def __init__(self, pre_trained=False, frozen=False):
        super(ResNet50FeatureExtractor3D, self).__init__()
        # Start with 2D ResNet as base, but we'll modify its layers
        resnet2d = models.resnet50(pretrained=pre_trained)
        
        # Convert 2D conv to 3D conv for the input layer (1 channel input for medical images)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        
        # Convert BatchNorm2d to BatchNorm3d
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Convert all layers from 2D to 3D
        self.layer1 = self._make_layer_3d(resnet2d.layer1, 64, 64, 3)
        self.layer2 = self._make_layer_3d(resnet2d.layer2, 256, 128, 4)
        self.layer3 = self._make_layer_3d(resnet2d.layer3, 512, 256, 6)
        self.layer4 = self._make_layer_3d(resnet2d.layer4, 1024, 512, 3)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Get the number of output features
        self.n_features_out = 2048  # Same as in ResNet50
        
        if frozen:
            for param in self.parameters():
                param.requires_grad = False
    
    def _make_layer_3d(self, layer_2d, in_channels, out_channels, blocks):
        """
        Convert a 2D ResNet layer to 3D
        
        Args:
            layer_2d: Original 2D layer from ResNet
            in_channels: Input channels for the first block
            out_channels: Output channels for the blocks
            blocks: Number of blocks in the layer
            
        Returns:
            nn.Sequential: 3D version of the layer
        """
        layers = []
        
        # Create equivalent 3D blocks based on the 2D architecture
        for i in range(blocks):
            # For the first block, we need to handle potential downsampling
            if i == 0:
                downsample = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels * 4, kernel_size=1, stride=2, bias=False),
                    # nn.BatchNorm3d(out_channels * 4)
                    nn.GroupNorm(min(32, out_channels * 4), out_channels * 4)
                )
                layers.append(Bottleneck3D(in_channels, out_channels, stride=2, downsample=downsample))
            else:
                layers.append(Bottleneck3D(out_channels * 4, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through 3D layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

class Bottleneck3D(nn.Module):
    """
    3D version of ResNet bottleneck block
    """
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        # 1x1x1 conv
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(min(32, planes), planes)
        # 3x3x3 conv
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(min(32, planes), planes)
        # 1x1x1 conv
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(min(32, planes * self.expansion), planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet50_3D(torch.nn.Module):
    def __init__(self, pre_trained=True, frozen=False):
        super(ResNet50_3D, self).__init__()
        self.cnn = ResNet50FeatureExtractor3D(pre_trained=pre_trained, frozen=frozen)
       
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.cnn.n_features_out, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(512, 1)
        )

    def forward(self, im):
        # im shape expected to be (batch_size, 1, 64, 64, 64)
        out = self.cnn.forward(im)
        out = self.classifier.forward(out)

        return out

class FeatureFusion3D(torch.nn.Module):
    def __init__(self, meta_features, pre_trained=False, frozen=False):
        super(FeatureFusion3D, self).__init__()

        self.meta_features = meta_features
        self.cnn = ResNet50FeatureExtractor3D(pre_trained=pre_trained, frozen=frozen)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.cnn.n_features_out + self.meta_features, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(512, 1)
        )

    def forward(self, im, meta):
        # im shape expected to be (batch_size, 1, 64, 64, 64)
        x1 = self.cnn.forward(im)
        x2 = meta

        x = torch.cat((x1, x2), dim=-1)
        x = self.classifier.forward(x)

        return x

class LearnedFeatureFusion3D(torch.nn.Module):
    def __init__(self, meta_features=None, mode="concat", pre_trained=False, frozen=False):
        super(LearnedFeatureFusion3D, self).__init__()
        assert mode in ["concat", "multiply", "add", "image_only"], "mode must be one of ['concat', 'multiply', 'add', 'image_only']"

        self.meta_features = meta_features
        self.mode = mode
        self.image_only = (mode == "image_only" or meta_features is None)

        # CNN backbone for image processing
        self.cnn = ResNet50FeatureExtractor3D(pre_trained=pre_trained, frozen=frozen)
        self.cnn.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(inplace=True)
        )
        
        # Only create metadata network if not in image_only mode
        if not self.image_only and meta_features is not None:
            self.meta_nn = torch.nn.Sequential(
                torch.nn.Linear(self.meta_features, 1024),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(inplace=True),
            )

        # Classifier with appropriate input size depending on mode
        classifier_input_size = 512*2 if (self.mode == "concat" and not self.image_only) else 512
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(classifier_input_size, 512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.25),
            torch.nn.Linear(512, 1)
        )

    def forward(self, im, meta=None):
        # Process image features
        x1 = self.cnn.forward(im)
        x1 = self.cnn.fc.forward(x1)
        
        # If image_only mode or no metadata provided, skip fusion
        if self.image_only or meta is None:
            x = x1
        else:
            # Process metadata
            x2 = self.meta_nn.forward(meta)
            
            # Apply selected fusion method
            if self.mode == "concat":
                x = torch.cat((x1, x2), dim=-1)
            elif self.mode == "multiply":
                x = x1 * x2
            elif self.mode == "add":
                x = x1 + x2
        
        # Final classification
        x = self.classifier.forward(x)
        
        return x

class ResNet18FeatureExtractor3D(torch.nn.Module):
    def __init__(self, pre_trained=False, frozen=False):
        super(ResNet18FeatureExtractor3D, self).__init__()
        # Start with 2D ResNet as base, but we'll modify its layers
        resnet2d = models.resnet50(pretrained=pre_trained)
        
        # Convert 2D conv to 3D conv for the input layer (1 channel input for medical images)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        
        # Convert BatchNorm2d to BatchNorm3d
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # Convert all layers from 2D to 3D
        self.layer1 = self._make_layer_3d(resnet2d.layer1, 64, 64, 2)
        self.layer2 = self._make_layer_3d(resnet2d.layer2, 64, 128, 2)
        self.layer3 = self._make_layer_3d(resnet2d.layer3, 128, 256, 2)
        self.layer4 = self._make_layer_3d(resnet2d.layer4, 256, 512, 2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Get the number of output features
        self.n_features_out = 512  # Same as in ResNet50
        
        if frozen:
            for param in self.parameters():
                param.requires_grad = False
    
    def _make_layer_3d(self, layer_2d, in_channels, out_channels, blocks):
        """
        Convert a 2D ResNet layer to 3D
        
        Args:
            layer_2d: Original 2D layer from ResNet
            in_channels: Input channels for the first block
            out_channels: Output channels for the blocks
            blocks: Number of blocks in the layer
            
        Returns:
            nn.Sequential: 3D version of the layer
        """
        layers = []
        
        # Create equivalent 3D blocks based on the 2D architecture
        for i in range(blocks):
            # For the first block, we need to handle potential downsampling
            if i == 0:
                downsample = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                    # nn.BatchNorm3d(out_channels * 4)
                    nn.GroupNorm(min(32, out_channels), out_channels)
                )
                layers.append(BasicBlock3D(in_channels, out_channels, stride=2, downsample=downsample))
            else:
                layers.append(BasicBlock3D(out_channels, out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through 3D layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out