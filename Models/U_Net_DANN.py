import torch.nn as nn
import torch
import torch.nn.functional as F
from Models.BuildingBlocks import *

class FE(nn.Module):
    """
      Class for the creation of the feature extractor.
    """
    def __init__(self, n_channels, starter, up_layer, bilinear = True):
        super(FE, self).__init__()

        self.n_channels = n_channels
        self.bilinear = bilinear
        self.starter = starter
        self.up_layer = up_layer

        # Layers related to segmentation task
        self.inc = (DoubleConv(self.n_channels, self.starter))
        self.down1 = (Down(self.starter, self.starter*(2**1)))
        self.down2 = (Down(self.starter*(2**1), self.starter*(2**2)))
        self.down3 = (Down(self.starter*(2**2), self.starter*(2**3)))
        factor = 2 if bilinear else 1
        self.down4 = (Down(self.starter*(2**3), self.starter*(2**4) // factor))
        if self.up_layer >= 1:
            self.up1 = (Up(self.starter*(2**4), self.starter*(2**3) // factor, bilinear))
        if self.up_layer >= 2:
            self.up2 = (Up(self.starter*(2**3), self.starter*(2**2) // factor, bilinear))
        if self.up_layer >= 3:
            self.up3 = (Up(self.starter*(2**2), self.starter*(2**1) // factor, bilinear))
        if self.up_layer >= 4:
            self.up4 = (Up(self.starter*(2**1), self.starter, bilinear))

    def DownSteps(self, x):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1, x2, x3, x4, x5

    def forward(self, x):

        # Downsample steps
        x1, x2, x3, x4, x5 = self.DownSteps(x)

        # Upsample steps
        if self.up_layer == 0:
            x = x5
        if self.up_layer >= 1:
            x = self.up1(x5, x4)
        if self.up_layer >= 2:
            x = self.up2(x, x3)
        if self.up_layer >= 3:
            x = self.up3(x, x2)
        if self.up_layer >= 4:
            x = self.up4(x, x1)

        return x

class C(nn.Module):
    def __init__(self, n_channels, starter, up_layer, bilinear = True, n_classes = 2):
        super(C, self).__init__()

        self.n_channels = n_channels
        self.bilinear = bilinear
        self.starter = starter
        self.up_layer = up_layer
        self.n_classes = n_classes

        factor = 2 if bilinear else 1

        if self.up_layer == 0:
            self.up1 = (Up(self.starter*(2**4), self.starter*(2**3) // factor, bilinear))
            self.up2 = (Up(self.starter*(2**3), self.starter*(2**2) // factor, bilinear))
            self.up3 = (Up(self.starter*(2**2), self.starter*(2**1) // factor, bilinear))
            self.up4 = (Up(self.starter*(2**1), self.starter, bilinear))
            self.outc = (OutConv(self.starter, n_classes))
        elif self.up_layer == 1:
            self.up2 = (Up(self.starter*(2**3), self.starter*(2**2) // factor, bilinear))
            self.up3 = (Up(self.starter*(2**2), self.starter*(2**1) // factor, bilinear))
            self.up4 = (Up(self.starter*(2**1), self.starter, bilinear))
            self.outc = (OutConv(self.starter, n_classes))
        elif self.up_layer == 2:
            self.up3 = (Up(self.starter*(2**2), self.starter*(2**1) // factor, bilinear))
            self.up4 = (Up(self.starter*(2**1), self.starter, bilinear))
            self.outc = (OutConv(self.starter, n_classes))
        elif self.up_layer == 3:
            self.up4 = (Up(self.starter*(2**1), self.starter, bilinear))
            self.outc = (OutConv(self.starter, n_classes))
        elif self.up_layer == 4:
            self.outc = (OutConv(self.starter, n_classes))


    def forward(self, x, dw):
        # Downsample steps
        x1, x2, x3, x4, x5 = dw

        # Upsampling steps
        if self.up_layer == 0:
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
        elif self.up_layer == 1:
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
        elif self.up_layer == 2:
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            logits = self.outc(x)
        elif self.up_layer == 3:
            x = self.up4(x, x1)
            logits = self.outc(x)
        elif self.up_layer == 4:
            logits = self.outc(x)

        return logits

class D(nn.Module):
    def __init__(self, initial_features, bilinear=True, starter = 8, up_layer = 3):
        super(D, self).__init__()

        self.initial_features = initial_features
        self.bilinear = bilinear
        self.starter = starter
        self.up_layer = up_layer

        factor = 2 if bilinear else 1

        self.revgrad = (GradientReversal())
        self.outd = (OutDisc(self.initial_features, 20))

        if self.up_layer > 0:
            self.down4_D = (Down(self.starter*(2**3)//factor, self.starter*(2**4) // factor))
        if self.up_layer > 1:
            self.down3_D = (Down(self.starter*(2**2)//factor, self.starter*(2**3)//factor))
        if self.up_layer > 2:
            self.down2_D = (Down(self.starter*(2**1)//factor, self.starter*(2**2)//factor))
        if self.up_layer > 3:
            self.down1_D = (Down(self.starter, self.starter*2//factor))

    def forward(self, x):

        x = self.revgrad(x)

        if self.up_layer == 1:
            x = self.down4_D(x)
        if self.up_layer == 2:
            x = self.down3_D(x)
            x = self.down4_D(x)
        if self.up_layer == 3:
            x = self.down2_D(x)
            x = self.down3_D(x)
            x = self.down4_D(x)
        if self.up_layer == 4:
            x = self.down1_D(x)
            x = self.down2_D(x)
            x = self.down3_D(x)
            x = self.down4_D(x)

        x = self.outd(x)

        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, starter = 8, up_layer = 3):

        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.starter = starter
        self.up_layer = up_layer

        self.FE = (FE(self.n_channels, self.starter, self.up_layer, self.bilinear))
        self.C = (C(self.n_channels, self.starter, self.up_layer, self.bilinear, self.n_classes))
        # self.D = (D(biliniear = self.bilinear, starter = self.starter, up_layer = self.up_layer))

        self.apply(self._init_weights)

    def forward(self, x):

        features = self.FE(x) # Feature extractor
        down_st = self.FE.DownSteps(x) # Get channels that will be concatenated from downward steps

        logits = self.C(features, down_st) # Classifier

        return logits


    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

class disc(nn.Module):
    def __init__(self, in_feat, bilinear=True, starter = 8, up_layer = 3):

        super(disc, self).__init__()

        self.bilinear = bilinear
        self.starter = starter
        self.up_layer = up_layer
        self.in_feat = in_feat

        self.D = (D(initial_features=self.in_feat, bilinear = self.bilinear, starter = self.starter, up_layer = self.up_layer))

    def forward(self, x):

        disc = self.D(x)

        return disc

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)