import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from ddg.utils import LOSS_REGISTRY

__all__ = ['cosine_pairwise_loss', 'cosine_loss',
           'orthogonal_projection_loss']


class CosineLoss(nn.Linear):
    r"""
    Cosine Loss
    """
    def __init__(self, in_features, out_features, bias=False):
        super(CosineLoss, self).__init__(in_features, out_features, bias)
        self.s_ = torch.nn.Parameter(torch.zeros(1))

    def loss(self, Z, target):
        s = F.softplus(self.s_).add(1.)
        l = F.cross_entropy(Z.mul(s), target, weight=None, ignore_index=-100, reduction='mean')
        return l
        
    def forward(self, input, target):
        logit = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1), self.bias) # [N x out_features]
        l = self.loss(logit, target)
        return logit, l


class tvMFLoss(CosineLoss):
    r"""
    t-vMF Loss
    """
    def __init__(self, in_features, out_features, bias=False, kappa=16):
        super(tvMFLoss, self).__init__(in_features, out_features, bias)
        self.register_buffer('kappa', torch.Tensor([kappa]))

    def forward(self, input, target=None):
        assert target is not None
        cosine = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1), None) # [N x out_features]
        logit =  (1. + cosine).div(1. + (1.-cosine).mul(self.kappa)) - 1.

        if self.bias is not None:
            logit.add_(self.bias)

        l = self.loss(logit, target)
        return logit, l
    
    def extra_repr(self):
        return super(tvMFLoss, self).extra_repr() + ', kappa={}'.format(self.kappa)


class CosinePairwiseLoss(nn.Module):
    r"""
    Cosine Pairwise Loss
    """
    def __init__(self):
        super(CosinePairwiseLoss, self).__init__()

    def forward(self, input):
        cs_matrix = torch.triu(pairwise_cosine_similarity(input))
        avg_cs = torch.sum(cs_matrix) / (input.shape[0] * input.shape[0] / 2)
        assert avg_cs <= 1 and avg_cs >= -1
        return 1 - avg_cs    


class OrthogonalLoss(nn.Module):
    r"""
    Orthogonal Loss
    """
    def __init__(self):
        super(OrthogonalLoss, self).__init__()

    def forward(self, input):
        return torch.norm(input.t() @ input - torch.eye(input.shape[1]), p='fro')




# https://github.com/kahnchana/opl/blob/master/loss.py
class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss


@LOSS_REGISTRY.register()
def cosine_pairwise_loss(logit):
    loss = CosinePairwiseLoss()(logit)
    return loss 

@LOSS_REGISTRY.register()
def cosine_loss(logit, target):
    loss = CosineLoss(logit.shape[1], target.max() + 1)(logit, target)
    return loss

@LOSS_REGISTRY.register()
def orthogonal_loss(common_feature, specific_feature):
    diag_tensor = torch.stack([torch.eye(logit.shape[1]) for _ in range(logit.shape[0])], dim=0).cuda()
    cps = torch.stack([torch.matmul(logit[:, :, _], torch.transpose(logit[:, :, _], 0, 1)) for _ in range(logit.shape[0])], dim=0)
    loss = OrthogonalLoss()(common_feature, specific_feature)
    return loss

@LOSS_REGISTRY.register()
def orthogonal_projection_loss(logit):
    loss = OrthogonalProjectionLoss()(logit)
    return loss