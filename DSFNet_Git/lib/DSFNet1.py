import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Res2Net_v1b import res2net50_v1b_26w_4s
#from .ResNet import ResNet
#import torchvision.models as models
from utils.tensor_ops import cus_sample, upsample_add
import torchvision.ops
from model.Deformable import DeCNet


#.................................. Separable Convolution Network .......................
class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class SCNet(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(SCNet, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)

#............. Basic Convolution Module ......................
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, relu=False, bn=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# ---------- New Deformable Separable Residual Fusion Block (DSRFB)  ------------------------
class DSRFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DSRFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            SCNet(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            SCNet(in_channel, out_channel, 1),
            DeCNet(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            DeCNet(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            SCNet(in_channel, out_channel, 1),
            DeCNet(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            DeCNet(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            SCNet(in_channel, out_channel, 1),
            DeCNet(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            DeCNet(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = DeCNet(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = SCNet(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)

        x = self.relu(x_cat + self.conv_res(x))
        #print(x.size())
        return x

# Deformable Region Context Module 
class DRCM(nn.Module):
    def __init__(self, channels=64, r=4):
        super(DRCM, self).__init__()
        out_channels = int(channels // r)
        # local_att
        self.local_att = nn.Sequential(
            DeCNet(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DeCNet(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AvgPool2d(1),
            DeCNet(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DeCNet(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )
        self.conv = nn.Conv2d(out_channels, channels, kernel_size=1, 1, 1)

        self.sig = nn.Sigmoid()

    def forward(self, x):

        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg
        xlg = self.conv(xlg)
        wei = self.sig(xlg)

        return wei

#....... Deformable Attention Fusion Module
class DAFM(nn.Module):
    def __init__(self, channel=64):
        super(DAFM, self).__init__()

        self.drcm = DRCM()
        self.upsample = cus_sample
        #self.down = BasicConv2d(64, 64, kernel_size=3, stride=1,padding=1,relu=True)
        self.dconv1 = DeCNet(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.dconv2 = DeCNet(channel, channel, kernel_size=3, stride=1, padding=1,relu=True)
        self.sconv =  SCNet(channel, channel, 1, 1)

    def forward(self, x, y):
        x = self.dconv1(x)
        y = self.dconv2(y)
        #y = self.upsample(y, scale_factor=2)
       # print(x.size())
       # print(y.size())
        #y = self.down(y)
       # print(y.size())
        xy = x + y
       # print(xy.size())
        wei = self.drcm(xy)
        wei = x * xy
        x = wei + self.upsample(y, scale_factor=1)
       # print(x.size())
        x0 = x * wei + y * (1 - wei)
        x0 = self.sconv(x0)

        return x0

# ........ Deformable Global Attention Module................
class DGAM(nn.Module):
    def __init__(self, channel=64):
        super(DGAM, self).__init__()
        self.h2l_pool = nn.AvgPool2d(1,stride=1)

        self.h2l = DeCNet(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)
        self.h2h = DeCNet(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

        self.drcmh = DRCM()
        self.drcml = DRCM()
        
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        self.dconv = DeCNet(channel, channel, kernel_size=1, stride=1, padding=1, relu=True)

    def forward(self, x):

        # 
        x_h = self.h2h(x)
        x_l = self.h2l(self.h2l_pool(x))
        #print(x_h.size())
        #print(x_l.size())
        x_h1 = self.drcmh(x_h)
        x_l1 = self.drcml(x_l)
       # print(x_h1.size())
        #x_l = self.upsample(x_l,scale_factor=2)
       # print(x_l1.size())
        #x_h = x_h1 * self.drcmh(x_h1)
        #x_l = x_l1 *  self.drcml(x_l1)
        x_h1 = x_h1 * F.interpolate(x_h1, scale_factor=1, mode='bilinear', align_corners=False)
        x_l1 = x_l1 * F.interpolate(x_l1, scale_factor=1, mode='bilinear', align_corners=False)
        x_h = x_h * x_l1#self.upsample(x_h,scale_factor=1)#F.interpolate(x_h, scale_factor=1, mode='bilinear', align_corners=False)
        x_l = x_l * x_h1#self.upsample(x_l,scale_factor=1)#F.interpolate(x_l, scale_factor=1, mode='bilinear', align_corners=False)
       # print(x_h.size())
       # print(x_l.size())

        out = self.upsample_add(x_l, x_h)
        out = self.dconv(out)

        return out

# Local Feature Indicator (LFI)        
class LFI(nn.Module):
    """
    the LFI is employed to refine the local feature and surrounding context.
    """
    def __init__(self, channel, reduction=16):
        super(LFI, self).__init__()
        self.scnet = SCNet(channel, channel, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
       # x = self.scnet(x)
        y = self.relu(self.avg_pool(x).view(b, c))
        y = self.fc(y).view(b, c, 64,64)
        out = self.sig(y)
        return out


#..... Deformable Separable Fusion Network.............
class DSFNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=64):
        super(DSFNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # ---- Deformable Separable Receptive Field Block like module ----
        self.dsrfb2_1 = DSRFB(channel, channel)
        self.dsrfb3_1 = DSRFB(channel, channel)
        self.dsrfb4_1 = DSRFB(channel, channel)
        #..... DAFM........
        self.dafm3 = DAFM()
        self.dafm2 = DAFM()
        #..... DGAM........
        self.dgam3 = DGAM()
        self.dgam2 = DGAM()
        #......... Local feature endicator.......
        self.lfi = LFI(64, 64)
        self.upsample = cus_sample
        #...... Downsampling the feature..............................
        self.down1 = nn.Conv2d(2048, 64, kernel_size=3, stride=1,padding=1)
        self.down2 = nn.Conv2d(512, 64, kernel_size=3, stride=1,padding=1)
        self.down3 = nn.Conv2d(256, 64, kernel_size=3, stride=1,padding=1)

        self.down4 = nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1)
        self.down5 = nn.Conv2d(1024, 64, kernel_size=3, stride=1,padding=1)
        
        self.upconv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        #....... Saliency map generation.........
        self.classifier = SCNet(192, 1, 1)
        self.upsample_add = upsample_add
        

    def forward(self, x):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 88, 88

        # ---- low-level backbone features ----
        x1 = self.resnet.layer1(x)  # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)  # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)  # bs, 2048, 11, 11
        
        # Local and Global Feature Extraction
        x4 = self.down1(x4)
        x2_rfb = self.dsrfb2_1(x4)  # channel -> 64
        x3_rfb = self.dsrfb3_1(x2_rfb)  # channel -> 64
        x4_rfb = self.dsrfb4_1(x3_rfb)  # channel -> 64
        
        # Local Feature Extraction 
        x2_rfb = F.interpolate(x2_rfb, scale_factor=8, mode='bilinear', align_corners=False)
        x2 = self.down2(self.upsample(x2,scale_factor=2))
        #print(x2_rfb.size())
        #print(x2.size())
        l = self.down4(x2 + x2_rfb)
        l = self.lfi(l)
        print(l.size())

        x4_rfb = F.interpolate(x4_rfb, scale_factor=8, mode='bilinear', align_corners=False)
        x1 = self.down3(x1)
        #print(x4_rfb.size())
        #print(x1.size())
        out42 = self.down4(x1 + x4_rfb)
        #print(l.size())
        #print(out42.size())
        x42 = self.dafm2(l, out42)

        #print(x42.size())
     
        out42 = self.upsample(x42 + self.dgam2(x42))

        # Global Feature Extraction 
        #print(x4.size())
        #print(x2_rfb.size())
        x4 = F.interpolate(x4, scale_factor=8, mode='bilinear', align_corners=False)
        x43 = self.down4(x4 + x2_rfb)
 
        x3_rfb = F.interpolate(x3_rfb, scale_factor=2, mode='bilinear', align_corners=False)
        #print(x43.size())
        #print(x3.size())
        #print(x3_rfb.size())
        x3 = self.down5(x3)
        #print(x3.size())
        x432 = x3 + x3_rfb
        #print(x3.size())
        #print(x432.size())
        x43 = self.dafm3(x43, x432)
        #out43 = self.dgam3(x43)
        #print(out43.size())
        out43 = self.upsample(x43 + self.dgam3(x43))
        
        #print(out43.size())
        out432 =  self.upsample_add(torch.cat((out42, out43, x4_rfb),dim=1)) 
        
        # Saliency map generation
        s3 = self.classifier(out432)
        s3 = F.interpolate(s3, scale_factor=4, mode='bilinear', align_corners=False)

        return s3
        

if __name__ == '__main__':
    ras = DSFNet().cuda()
    input_tensor = torch.randn(2, 3, 352, 352).cuda()

    out = ras(input_tensor)
