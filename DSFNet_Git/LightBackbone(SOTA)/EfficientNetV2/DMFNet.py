import torch
import torch.nn as nn
#from model.HolisticAttention import HA
#from lib.vgg import VGG
from EfficientNetV2 import EfficientNet
from utils.tensor_ops import cus_sample, upsample_add
import torchvision.ops
import torch.nn.functional as F
from torch.nn import Softmax
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from random import randrange


MODEL_MAP = {
    'efficientdet-d0': 'efficientnet-b0',
    'efficientdet-d1': 'efficientnet-b1',
    'efficientdet-d2': 'efficientnet-b2',
    'efficientdet-d3': 'efficientnet-b3',
    'efficientdet-d4': 'efficientnet-b4',
    'efficientdet-d5': 'efficientnet-b5',
    'efficientdet-d6': 'efficientnet-b6',
    'efficientdet-d7': 'efficientnet-b6',
}

def exists(val):
    return val is not None

def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) < dropout

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

class DeCNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super(DeCNet, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x) 
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        # op = (n - (k * d - 1) + 2p / s)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

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

class DSConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv2d, self).__init__()
        self.conv = nn.Sequential(
                convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
                convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
                )

    def forward(self, x):
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

#............. Attention...................


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

#................... Deformable Cross Attention Module (DCAM)......
class DCAM(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(DCAM,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x

#............. DAEM.........................
class DAEM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DAEM, self).__init__()
        self.relu = nn.ReLU(False)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            BasicConv2d(out_channel, out_channel, 1, padding=0),
          #  nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False))

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            BasicConv2d(out_channel, out_channel, 1, padding=0),
       #     BasicConv2d(out_channel, out_channel, 1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False))

        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            BasicConv2d(out_channel, out_channel, 1, padding=0),
            nn.BatchNorm2d(out_channel),
            BasicConv2d(out_channel, out_channel, 1, padding=0),
            #nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False))

        self.sa = DCAM(out_channel)
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 1, padding=0)
        self.conv_res = BasicConv2d(out_channel, out_channel, 1, padding=0)

    def forward(self, x):
        x0 = self.branch0(x)
        x0 = self.sa(x0)
        #print(x0.size())
        x1 = self.branch1(x)
        x1 = self.sa(x1)
        #print(x1.size())
        x2 = self.branch2(x)
        x2 = self.sa(x2)
        #print(x2.size())
        x3 = self.branch3(x)
        x3 = self.sa(x3)
        #print(x3.size())
        #x4 = self.branch4(x)
        #x4 = self.sa(x4)
        #print(x4.size())
        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)
        #print(x_cat.size())
        #print(x.size())
        x = self.relu(x_cat + self.conv_res(x))
        #print(x.size())
        return x
#......................... RFB................

class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x
        
#..................... LayerNorm ............................
class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b    

#............... Multi-scale deformable Fusion Network.............
class DFNet(nn.Module):
     def __init__(self, channel):
         super(DFNet, self).__init__()
         self.relu = nn.ReLU(True)

         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
         self.upsample1 = cus_sample

         self.decoder5 = nn.Sequential(
             BasicConv2d(64, channel, 3,  padding=1),
             LayerNorm(channel),
             nn.Dropout(0.5))

         self.S5 = BasicConv2d(channel, 1, 1)#, stride=1, padding=1)

         self.decoder4 = nn.Sequential(
             BasicConv2d(128, channel, 3, padding=1),
             LayerNorm(channel),
             nn.Dropout(0.5))

         self.S4 = BasicConv2d(channel, 1, 1)#, stride=1, padding=1)

         self.decoder3 = nn.Sequential(
             BasicConv2d(128, channel, 3, padding=1),
             LayerNorm(channel),
             nn.Dropout(0.5))

         self.S3 = BasicConv2d(channel, 1, 1)#, stride=1, padding=1)

         self.decoder2 = nn.Sequential(
             BasicConv2d(128, channel, 3, padding=1),
             LayerNorm(channel),
             nn.Dropout(0.5))

         self.S2 = BasicConv2d(channel, 1, 1)

     def forward(self, x5, x4, x3, x2):

         # x5: 1/16, 512; x4: 1/8, 512; x3: 1/4, 256; x2: 1/2, 128; x1: 1/1, 64
         x5_up = self.decoder5(x5)
         s5 = self.S5(x5_up)
         #print(x5.size())
         x5_up = self.upsample1(x5_up, scale_factor=2)
         #print(x5_up.size())
         #x4 = self.upsample1(x4, scale_factor=2)
         #print(x4.size())
         x4_up = self.decoder4(torch.cat((x4, x5_up), 1))
         x4_up = self.upsample1(x4_up, scale_factor=2)
         #print(x4_up.size())
         s4 = self.S4(x4_up)
         #print(x3.size())
         #print(s4.size())
         x3_up = self.decoder3(torch.cat((x3, x4_up), 1))
         x3_up = self.upsample1(x3_up, scale_factor=2)
         #print(x3_up.size())
         s3 = self.S3(x3_up)
         #print(x2.size())
         x2_up = self.decoder2(torch.cat((x2, x3_up), 1))
         #x2_up = self.upsample1(x2_up, scale_factor=2)
         #print(x2_up.size())
         s2 = self.S2(x2_up)
         #print(s2.size())
        #  x1_up = self.decoder1(torch.cat((x1, x2_up), 1))
        #  s1 = self.S1(x1_up)
         #print(s1.size())
         return  s2, s3, s4, s5


#----------------- DMFNet------------------------------
class DMFNet(nn.Module):
    def __init__(self, network='efficientdet-d0'):
        super(DMFNet, self).__init__()
        #Backbone model
        #self.vgg = VGG('rgb')
        self.vgg = EfficientNet.from_pretrained(f'efficientnet-b0', advprop=True)#EfficientNet.from_pretrained(MODEL_MAP[network])
        #............. Encoder...........

        self.e2 = nn.Sequential(BasicConv2d(64, 64, 3, 1, 1),BasicConv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.e3 = nn.Sequential(BasicConv2d(64, 64, 3, 1, 1),BasicConv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.e4 = nn.Sequential(BasicConv2d(64, 64, 3, 1, 1),BasicConv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.e5 = nn.Sequential(BasicConv2d(64, 64, 3, 1, 1),BasicConv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        
        #............. DAEM .................

        self.daem_5 = DAEM(64, 64)
        self.daem_4 = DAEM(64, 64)
        self.daem_3 = DAEM(64, 64)
        self.daem_2 = DAEM(64, 64)
        
        #.............. RFB ..............

        self.rfb2 = RFB(24, 64)
        self.rfb3 = RFB(40, 64)
        self.rfb4 = RFB(112, 64)
        self.rfb5 = RFB(320, 64)
        #............... Decoder...........

        self.d2 = nn.Sequential(BasicConv2d(64, 64, 3, 1, 1),BasicConv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64),  nn.ReLU())
        self.d3 = nn.Sequential(BasicConv2d(64, 64, 3, 1, 1),BasicConv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64),  nn.ReLU())
        self.d4 = nn.Sequential(BasicConv2d(64, 64, 3, 1, 1),BasicConv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64),  nn.ReLU())
        self.d5 = nn.Sequential(BasicConv2d(64, 64, 3, 1, 1),BasicConv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64),  nn.ReLU()) 
        
        #.... Multi-domain fusion decoder....

        self.dfnet_rgb = DFNet(64)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.sigmoid = nn.Sigmoid()
        #self.att = Attention(64)
        self.upsample = cus_sample

    def forward(self, x_rgb):
        B, C, H, W = x_rgb.size()

        # EfficientNet backbone Encoder
        x = self.vgg.initial_conv(x_rgb)
        features = self.vgg.get_blocks(x, H, W)
        #.......... Backbone Network..
        #features = self.vgg(x_rgb)
        x2_rgb = features[0]
        x3_rgb = features[1]
        x4_rgb = features[2]
        x5_rgb = features[3]
        #print(x2_rgb.size())
        #print(x3_rgb.size())
        #print(x4_rgb.size())
        #rint(x5_rgb.size())
        # x1_rgb = self.vgg.conv1(x_rgb)
        # x2_rgb = self.vgg.conv2(x1_rgb)
        # x3_rgb = self.vgg.conv3(x2_rgb)
        # x4_rgb = self.vgg.conv4(x3_rgb)
        # x5_rgb = self.vgg.conv5(x4_rgb)
        #print(x1_rgb.size())
        #................ RFB .........

        x2_rgb = self.rfb2(x2_rgb)
        x3_rgb = self.rfb3(x3_rgb)
        x4_rgb = self.rfb4(x4_rgb)
        x5_rgb = self.rfb5(x5_rgb)

        #................ Encoder .....

        x2_e = self.e2(x2_rgb)
        x3_e = self.e3(x3_rgb)
        x4_e = self.e4(x4_rgb)
        x5_e = self.e5(x5_rgb)

        #............... Decoder ........

        #x1_d = self.d1(x1_rgb)
        x2_d = self.d2(x2_rgb)
        x3_d = self.d3(x3_rgb)
        x4_d = self.d4(x4_rgb)
        x5_d = self.d5(x5_rgb)

        #... daem means enhance ..

        x5_en = self.daem_5(x5_rgb)
        x4_en = self.daem_4(x4_rgb)
        x3_en = self.daem_3(x3_rgb)
        x2_en = self.daem_2(x2_rgb)
        #print(x5_en.size())

        #.... Cross-correlation Fusion module....

        x5_en = x5_e * x5_en * x5_d
        #print(x5_en.size())
        x4_en = x4_e * x4_en * x4_d
        #print(x4_en.size())
        x3_en = x3_e * x3_en * x3_d
        #print(x3_en.size())
        x2_en = x2_e * x2_en * x2_d
        #print(x2_en.size())
      #  x1_en = x1_e * x1_en * x1_d
        #print(x1_en.size())

        #...... Multi-domain fusion decoder .......

        s2, s3, s4 ,s5 = self.dfnet_rgb(x5_en, x4_en, x3_en, x2_en)

        #...... Upsampling Operation .....
        s2 = self.upsample(s2, scale_factor=4)
        s3 = self.upsample(s3, scale_factor=4)
        s4 = self.upsample(s4,scale_factor=8)
        s5 = self.upsample(s5, scale_factor=32)
        #print(s2.size())
        #print(s3.size())
        #print(s4.size())
        #print(s5.size())

        return s2, s3, s4, s5,  self.sigmoid(s2), self.sigmoid(s3), self.sigmoid(s4), self.sigmoid(s5)
