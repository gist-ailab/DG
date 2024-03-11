import cv2
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# from util.utils import *

def GMM(feat, vecs, pred, true_mask, cls_label):
    b, k, oh, ow = pred.size()

    preserve = (true_mask < 255).long().view(b, 1, oh, ow)
    preserve = F.interpolate(preserve.float(), size=feat.size()[-2:], mode='bilinear')
    pred = F.interpolate(pred, size=feat.size()[-2:], mode='bilinear')
    _, _, h, w = pred.size()

    vecs = vecs.view(b, k, -1, 1, 1)
    feat = feat.view(b, 1, -1, h, w)

    """ 255 caused by cropping, using preserve mask """
    abs = torch.abs(feat - vecs).mean(2)
    abs = abs * cls_label.view(b, k, 1, 1) * preserve.view(b, 1, h, w)
    abs = abs.view(b, k, h*w)

    # """ calculate std """
    # pred = pred * preserve
    # num = pred.view(b, k, -1).sum(-1)
    # std = ((pred.view(b, k, -1)*(abs ** 2)).sum(-1)/(num + 1e-6)) ** 0.5
    # std = std.view(b, k, 1, 1).detach()

    # std = ((abs ** 2).sum(-1)/(preserve.view(b, 1, -1).sum(-1)) + 1e-6) ** 0.5
    # std = std.view(b, k, 1, 1).detach()

    abs = abs.view(b, k, h, w)
    res = torch.exp(-(abs * abs))
    # res = torch.exp(-(abs*abs)/(2*std*std + 1e-6))
    res = F.interpolate(res, size=(oh, ow), mode='bilinear')
    res = res * cls_label.view(b, k, 1, 1)

    return res
