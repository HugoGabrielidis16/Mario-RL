import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic ResNet block for ResNet-18 and ResNet-34"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out, inplace=True)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-50, ResNet-101, ResNet-152"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out, inplace=True)

        return out


class MultiFrameResNet(nn.Module):
    """
    ResNet model adapted for multi-frame RL inputs (frame stacking)
    
    Supports different fusion strategies for temporal information:
    - 'early': Fuse frames at input level (recommended)
    - 'late': Process frames separately then fuse
    - 'conv3d': Use 3D convolutions for temporal modeling
    """
    
    def __init__(self, 
                 n_actions=3,
                 state_shape=(4, 64, 64),  # (frame_stack, height, width)
                 layers=[2, 2, 2, 2], 
                 block=BasicBlock,
                 fusion_strategy='early',
                 dropout_rate=0.0,
                 *args,
                 **kwargs):
        super(MultiFrameResNet, self).__init__()
        
        # Parse state shape
        if len(state_shape) == 3:
            self.frame_stack, self.height, self.width = state_shape
        else:
            raise ValueError(f"Expected state_shape (frame_stack, H, W), got {state_shape}")
        
        self.n_actions = n_actions
        self.fusion_strategy = fusion_strategy
        self.current_channels = 64
        
        print(f"🧠 Initializing MultiFrameResNet:")
        print(f"   • Frame stack: {self.frame_stack}")
        print(f"   • Input size: {self.height}x{self.width}")
        print(f"   • Fusion strategy: {fusion_strategy}")
        print(f"   • Actions: {n_actions}")
        
        if fusion_strategy == 'early':
            self._build_early_fusion(block, layers, dropout_rate)
        elif fusion_strategy == 'late':
            self._build_late_fusion(block, layers, dropout_rate)
        elif fusion_strategy == 'conv3d':
            self._build_conv3d_fusion(layers, dropout_rate)
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        # Initialize weights
        self._initialize_weights()

    def _build_early_fusion(self, block, layers, dropout_rate):
        """Early fusion: Treat stacked frames as multi-channel input"""
        
        # Initial convolution - input channels = frame_stack
        self.conv1 = nn.Conv2d(self.frame_stack, 64, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Standard ResNet layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Add dropout for regularization
        feature_size = 512 * block.expansion
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
            self.fc = nn.Linear(feature_size, self.n_actions)
        else:
            self.dropout = None
            self.fc = nn.Linear(feature_size, self.n_actions)

    def _build_late_fusion(self, block, layers, dropout_rate):
        """Late fusion: Process each frame separately, then combine"""
        
        # Create separate CNN for each frame to avoid channel conflicts
        self.frame_cnns = nn.ModuleList()
        
        for i in range(self.frame_stack):
            # Create individual ResNet for each frame
            frame_cnn = nn.Sequential()
            
            # Initial layers
            frame_cnn.add_module('conv1', nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False))
            frame_cnn.add_module('bn1', nn.BatchNorm2d(64))
            frame_cnn.add_module('relu', nn.ReLU(inplace=True))
            frame_cnn.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            
            # ResNet layers - create fresh layers for each frame
            current_channels = 64
            
            # Layer 1
            layer1_modules = []
            for j in range(layers[0]):
                if j == 0:
                    layer1_modules.append(block(current_channels, 64))
                else:
                    layer1_modules.append(block(64 * block.expansion, 64))
            current_channels = 64 * block.expansion
            frame_cnn.add_module('layer1', nn.Sequential(*layer1_modules))
            
            # Layer 2
            layer2_modules = []
            for j in range(layers[1]):
                if j == 0:
                    downsample = None
                    if current_channels != 128 * block.expansion:
                        downsample = nn.Sequential(
                            nn.Conv2d(current_channels, 128 * block.expansion, kernel_size=1, stride=2, bias=False),
                            nn.BatchNorm2d(128 * block.expansion)
                        )
                    layer2_modules.append(block(current_channels, 128, stride=2, downsample=downsample))
                    current_channels = 128 * block.expansion
                else:
                    layer2_modules.append(block(current_channels, 128))
            frame_cnn.add_module('layer2', nn.Sequential(*layer2_modules))
            
            # Layer 3
            layer3_modules = []
            for j in range(layers[2]):
                if j == 0:
                    downsample = None
                    if current_channels != 256 * block.expansion:
                        downsample = nn.Sequential(
                            nn.Conv2d(current_channels, 256 * block.expansion, kernel_size=1, stride=2, bias=False),
                            nn.BatchNorm2d(256 * block.expansion)
                        )
                    layer3_modules.append(block(current_channels, 256, stride=2, downsample=downsample))
                    current_channels = 256 * block.expansion
                else:
                    layer3_modules.append(block(current_channels, 256))
            frame_cnn.add_module('layer3', nn.Sequential(*layer3_modules))
            
            # Layer 4
            layer4_modules = []
            for j in range(layers[3]):
                if j == 0:
                    downsample = None
                    if current_channels != 512 * block.expansion:
                        downsample = nn.Sequential(
                            nn.Conv2d(current_channels, 512 * block.expansion, kernel_size=1, stride=2, bias=False),
                            nn.BatchNorm2d(512 * block.expansion)
                        )
                    layer4_modules.append(block(current_channels, 512, stride=2, downsample=downsample))
                    current_channels = 512 * block.expansion
                else:
                    layer4_modules.append(block(current_channels, 512))
            frame_cnn.add_module('layer4', nn.Sequential(*layer4_modules))
            
            # Global average pooling
            frame_cnn.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
            
            self.frame_cnns.append(frame_cnn)
        
        # Fusion layer
        feature_size = 512 * block.expansion * self.frame_stack
        self.fusion = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        
        self.fc = nn.Linear(512, self.n_actions)

    def _build_conv3d_fusion(self, layers, dropout_rate):
        """3D convolution fusion: Explicitly model temporal relationships"""
        
        # 3D convolutions for temporal modeling
        self.conv3d_1 = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), 
                                 stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.bn3d_1 = nn.BatchNorm3d(64)
        
        # Reduce temporal dimension
        self.conv3d_2 = nn.Conv3d(64, 64, kernel_size=(self.frame_stack, 1, 1), 
                                 stride=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.bn3d_2 = nn.BatchNorm3d(64)
        
        # Standard 2D ResNet layers after temporal fusion
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
            
        self.fc = nn.Linear(512, self.n_actions)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.current_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.current_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.current_channels, out_channels, stride, downsample))
        self.current_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.current_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass for multi-frame input
        
        Args:
            x: Input tensor of shape (batch_size, frame_stack, height, width)
            
        Returns:
            Action values of shape (batch_size, n_actions)
        """
        batch_size = x.size(0)
        
        if self.fusion_strategy == 'early':
            return self._forward_early_fusion(x)
        elif self.fusion_strategy == 'late':
            return self._forward_late_fusion(x, batch_size)
        elif self.fusion_strategy == 'conv3d':
            return self._forward_conv3d_fusion(x, batch_size)

    def _forward_early_fusion(self, x):
        """Early fusion forward pass"""
        # Input: (batch_size, frame_stack, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if self.dropout is not None:
            x = self.dropout(x)
            
        x = self.fc(x)
        return x

    def _forward_late_fusion(self, x, batch_size):
        """Late fusion forward pass"""
        # Process each frame separately
        frame_features = []
        for i in range(self.frame_stack):
            frame = x[:, i:i+1, :, :]  # (batch_size, 1, H, W)
            features = self.frame_cnns[i](frame)
            features = torch.flatten(features, 1)
            frame_features.append(features)
        
        # Concatenate all frame features
        x = torch.cat(frame_features, dim=1)
        x = self.fusion(x)
        x = self.fc(x)
        return x

    def _forward_conv3d_fusion(self, x, batch_size):
        """3D convolution fusion forward pass"""
        # Reshape for 3D conv: (batch_size, 1, frame_stack, H, W)
        x = x.unsqueeze(1)
        
        x = self.conv3d_1(x)
        x = self.bn3d_1(x)
        x = F.relu(x, inplace=True)
        
        x = self.conv3d_2(x)
        x = self.bn3d_2(x)
        x = F.relu(x, inplace=True)
        
        # Remove temporal dimension: (batch_size, 64, 1, H, W) -> (batch_size, 64, H, W)
        x = x.squeeze(2)
        
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if self.dropout is not None:
            x = self.dropout(x)
            
        x = self.fc(x)
        return x

    def get_feature_size(self):
        """Get the size of features before the final layer"""
        if self.fusion_strategy == 'early':
            return 512 * BasicBlock.expansion
        elif self.fusion_strategy == 'late':
            return 512
        elif self.fusion_strategy == 'conv3d':
            return 512
        

# Factory functions for different ResNet variants
def resnet18_multiframe(n_actions, state_shape, fusion_strategy='early', **kwargs):
    """ResNet-18 for multi-frame RL"""
    return MultiFrameResNet(n_actions=n_actions, state_shape=state_shape,
                           layers=[2, 2, 2, 2], block=BasicBlock,
                           fusion_strategy=fusion_strategy, **kwargs)

def resnet34_multiframe(n_actions, state_shape, fusion_strategy='early', **kwargs):
    """ResNet-34 for multi-frame RL"""
    return MultiFrameResNet(n_actions=n_actions, state_shape=state_shape,
                           layers=[3, 4, 6, 3], block=BasicBlock,
                           fusion_strategy=fusion_strategy, **kwargs)

def resnet50_multiframe(n_actions, state_shape, fusion_strategy='early', **kwargs):
    """ResNet-50 for multi-frame RL"""
    return MultiFrameResNet(n_actions=n_actions, state_shape=state_shape,
                           layers=[3, 4, 6, 3], block=Bottleneck,
                           fusion_strategy=fusion_strategy, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    # Test different fusion strategies
    strategies = ['early', 'late', 'conv3d']
    state_shape = (4, 64, 64)  # 4 stacked frames, 64x64 pixels
    n_actions = 3
    batch_size = 8
    
    for strategy in strategies:
        print(f"\n🧪 Testing {strategy} fusion strategy:")
        
        model = resnet18_multiframe(
            n_actions=n_actions,
            state_shape=state_shape,
            fusion_strategy=strategy,
            dropout_rate=0.1
        )
        
        # Test with sample input
        sample_input = torch.randn(batch_size, *state_shape)
        
        # Forward pass
        output = model(sample_input)
        
        print(f"   ✅ Input shape: {sample_input.shape}")
        print(f"   ✅ Output shape: {output.shape}")
        print(f"   ✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test gradient flow
        loss = output.sum()
        loss.backward()
        print(f"   ✅ Gradient test: Passed")
        
    print(f"\n🎉 All tests passed!")