import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from utils.utils import clip_gradient, adjust_lr
import pytorch_iou
import numpy as np
import os, argparse
from datetime import datetime
from utils.data import get_loader
from utils.func import label_edge_prediction, AvgMeter
from lib.DMFNet import DMFNet
from loss import STandNST

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30, help='epoch number')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='batch size')
parser.add_argument('--trainsize', type=int, default=352, help='input size')
parser.add_argument('--trainset', type=str, default='Train', help='training  dataset')
opt = parser.parse_args()

# data preparing, set your own data path here
data_path = '/content/drive/MyDrive/data/DAVIS/'
image_root = data_path + opt.trainset + '/RGB/'
gt_root = data_path + opt.trainset + '/GT/'
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

#.......................... Loss..............
def structure_loss(pred, mask):
    w1 = torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
    w2 = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    w3 = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    weit = 1 + 0.5 * (w1 + w2 + w3) * mask
    #weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

# build models
model = DMFNet()
model.cuda()
params = model.parameters()
optimizer = torch.optim.SGD(params, opt.lr, momentum=0.9, weight_decay=5e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
CE = torch.nn.BCEWithLogitsLoss()
st = STandNST()
IOU = pytorch_iou.IOU(size_average = True)
size_rates = [0.75, 1, 1.25]  # multi-scale training
total_params = sum(
	param.numel() for param in model.parameters()
)
print(total_params)
# training
for epoch in range(0, opt.epoch):
    scheduler.step()
    model.train()
    loss_record1, loss_record2 = AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()

            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            # multi-scale training samples
            trainsize = int(round(opt.trainsize*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
              #  gt_edges = F.upsample(gt_edges, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # forward
            s2, s3, s4, s5, s2_sig, s3_sig, s4_sig, s5_sig = model(images)
            #loss1 = CE(s1, gts) + IOU(s1_sig, gts) + st(s1, gts)
            loss2 = structure_loss(s2, gts) + structure_loss(s2_sig, gts) + st(s2, gts)
            loss3 = structure_loss(s3, gts) + structure_loss(s3_sig, gts) + st(s3, gts)
            loss4 = structure_loss(s4, gts) + structure_loss(s4_sig, gts) + st(s4, gts)
            loss5 = structure_loss(s5, gts) + structure_loss(s5_sig, gts) + st(s5, gts)
            loss = loss2 + loss3 + loss4 + loss5

            loss.backward()

            optimizer.step()
            if rate == 1:
                loss_record1.update(loss4.data, opt.batchsize)
                loss_record2.update(loss5.data, opt.batchsize)

        if i % 1000 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f}, Loss2: {:.4f}'.
                format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record1.show(), loss_record2.show()))

save_path = './models/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
torch.save(model.state_dict(), save_path + opt.trainset + 'DSFNet.pth')
