"""
CNN models for stock price prediction from candlestick chart images.

Implements three CNN architectures:
- CNN5d: 5-day window (32x15 images)
- CNN20d: 20-day window (64x60 images)
- CNN60d: 60-day window (96x180 images)

Core concept: Each day is represented by 3 pixels: Open|High-Low bar|Close
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class CNN5d(nn.Module):
    """
    CNN model for 5-day candlestick chart analysis.
    
    Input: [batch_size, 1, 32, 15] - 32px height x 15px width (5 days x 3 pixels)
    Output: [batch_size, 1] - probability of price increase
    
    Architecture: 2 conv blocks + fully connected layer
    """
    
    def init_weights(self, m):
        """
        Initialize weights using Xavier initialization.
        Bias initialized to zero to prevent sigmoid saturation.
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.zero_()
    
    def __init__(self):
        super(CNN5d, self).__init__()
        
        # First conv block: basic feature extraction
        self.conv1 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(1, 64, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))),
            ('BN', nn.BatchNorm2d(64, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1)))
        ]))
        self.conv1 = self.conv1.apply(self.init_weights)
        
        # Second conv block: higher-level feature extraction
        self.conv2 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(64, 128, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))),
            ('BN', nn.BatchNorm2d(128, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1)))
        ]))
        self.conv2 = self.conv2.apply(self.init_weights)

        # Classification layer
        self.DropOut = nn.Dropout(p=0.5)
        self.FC = nn.Linear(15360, 1)  # 128×8×15 = 15,360 -> 1 output
        self.init_weights(self.FC)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass: candlestick image -> probability of price increase
        
        Args:
            x: Input image [batch_size, 32, 15]
            
        Returns:
            Probability [batch_size, 1] (0=down, 1=up)
        """
        # Add channel dimension: [N, 32, 15] -> [N, 1, 32, 15]
        x = x.unsqueeze(1).to(torch.float32)
        
        # Feature extraction pipeline
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Classification
        x = self.DropOut(x.view(x.shape[0], -1))
        x = self.FC(x)
        x = self.Sigmoid(x)
        
        return x
    
    
    
class CNN20d(nn.Module):
    """
    CNN model for 20-day candlestick chart analysis.
    
    Input: [batch_size, 1, 64, 60] - 64px height x 60px width (20 days x 3 pixels)
    Output: [batch_size, 1] - probability of price increase
    
    Architecture: 3 conv blocks + fully connected layer
    """
    
    def init_weights(self, m):
        """
        Initialize weights using Xavier initialization.
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def __init__(self):
        super(CNN20d, self).__init__()
        
        # First conv block: coarse pattern detection
        self.conv1 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(1, 64, (5, 3), padding=(3, 1), stride=(3, 1), dilation=(2, 1))),
            ('BN', nn.BatchNorm2d(64, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1)))
        ]))
        self.conv1 = self.conv1.apply(self.init_weights)
        
        # Second conv block: intermediate feature extraction
        self.conv2 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(64, 128, (5, 3), padding=(3, 1), stride=(1, 1), dilation=(1, 1))),
            ('BN', nn.BatchNorm2d(128, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1)))
        ]))
        self.conv2 = self.conv2.apply(self.init_weights)
        
        # Third conv block: fine-grained high-level feature extraction
        self.conv3 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(128, 256, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))),
            ('BN', nn.BatchNorm2d(256, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1)))
        ]))
        self.conv3 = self.conv3.apply(self.init_weights)

        # Classification layer
        self.DropOut = nn.Dropout(p=0.5)
        self.FC = nn.Linear(46080, 1)  # 256×3×60 = 46,080 -> 1 output
        self.init_weights(self.FC)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass: 20-day candlestick image -> probability of price increase
        
        Args:
            x: Input image [batch_size, 64, 60]
            
        Returns:
            Probability [batch_size, 1] (0=down, 1=up)
        """
        # Add channel dimension: [N, 64, 60] -> [N, 1, 64, 60]
        x = x.unsqueeze(1).to(torch.float32)
        
        # 3-stage feature extraction pipeline
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Classification
        x = self.DropOut(x.view(x.shape[0], -1))
        x = self.FC(x)
        x = self.Sigmoid(x)
        
        return x


class CNN60d(nn.Module):
    """
    CNN model for 60-day candlestick chart analysis.
    
    Input: [batch_size, 1, 96, 180] - 96px height x 180px width (60 days x 3 pixels)
    Output: [batch_size, 1] - probability of price increase
    
    Architecture: 4 conv blocks + fully connected layer
    """
    
    def init_weights(self, m):
        """
        Initialize weights using Xavier initialization.
        """
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def __init__(self):
        super(CNN60d, self).__init__()
        
        # First conv block: coarse pattern detection
        self.conv1 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(1, 64, (5, 3), padding=(2, 1), stride=(3, 1), dilation=(3, 1))),
            ('BN', nn.BatchNorm2d(64, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1)))
        ]))
        self.conv1 = self.conv1.apply(self.init_weights)
        
        # Second conv block: intermediate feature extraction
        self.conv2 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(64, 128, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))),
            ('BN', nn.BatchNorm2d(128, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1)))
        ]))
        self.conv2 = self.conv2.apply(self.init_weights)
        
        # Third conv block: high-level feature extraction
        self.conv3 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(128, 256, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))),
            ('BN', nn.BatchNorm2d(256, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1)))
        ]))
        self.conv3 = self.conv3.apply(self.init_weights)
        
        # Fourth conv block: highest-level complex pattern extraction
        self.conv4 = nn.Sequential(OrderedDict([
            ('Conv', nn.Conv2d(256, 512, (5, 3), padding=(2, 1), stride=(1, 1), dilation=(1, 1))),
            ('BN', nn.BatchNorm2d(512, affine=True)),
            ('LeakyReLU', nn.LeakyReLU()),
            ('Max-Pool', nn.MaxPool2d((2,1)))
        ]))
        self.conv4 = self.conv4.apply(self.init_weights)

        # Classification layer
        self.DropOut = nn.Dropout(p=0.5)
        self.FC = nn.Linear(184320, 1)  # 512×2×180 = 184,320 -> 1 output
        self.init_weights(self.FC)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass: 60-day candlestick image -> probability of price increase
        
        Args:
            x: Input image [batch_size, 96, 180]
            
        Returns:
            Probability [batch_size, 1] (0=down, 1=up)
        """
        # Add channel dimension: [N, 96, 180] -> [N, 1, 96, 180]
        x = x.unsqueeze(1).to(torch.float32)
        
        # 4-stage feature extraction pipeline
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Classification
        x = self.DropOut(x.view(x.shape[0], -1))
        x = self.FC(x)
        x = self.Sigmoid(x)
        
        return x
    