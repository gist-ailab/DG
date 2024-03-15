import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from ddg.utils import LOSS_REGISTRY

__all__ = ['cosine_pairwise_loss', 'cosine_loss',
           'orthogonal_projection_loss', 'orthogonal_loss']


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

    def forward(self, feature, pred):
        # feature.require_grad = True
        # cls_sum=[]
        cls_sum = 0
        for i in list(torch.unique(pred)):
            idx = torch.where(pred == i)
    
            if len(idx[0]) == 1:
                continue
            else: 
                # idx = idx.nonzero().squeeze(-1)
            # print('idx:', idx.shape)
                cls_feature = feature[idx[0]]
                
            # print('cls_feature:', cls_feature.shape)
            # print(pairwise_cosine_similarity(cls_feature).shape)
                # cs_matrix = torch.triu(pairwise_cosine_similarity(cls_feature))
                # avg_cs = torch.sum(cs_matrix) / (cls_feature.shape[0] * cls_feature.shape[0] / 2)
                cos_sim_matrix = pairwise_cosine_similarity(cls_feature).float()
                lower_tri_mask = torch.tril(torch.ones_like(cos_sim_matrix), diagonal=-1)
                avg_cs = torch.mean(cos_sim_matrix[lower_tri_mask.bool()])
                # avg_cs = torch.mean(pairwise_cosine_similarity(cls_feature).float())
                # print('fffffff', pairwise_cosine_similarity(cls_feature).shape)
            # assert avg_cs <= 1 and avg_cs >= -1
                # print(avg_cs.shape)
                # print(avg_cs)
                cls_sum = cls_sum + avg_cs
                # print(f'f_{avg_cs.shape}')
        # if len(cls_sum) == 0: 
        #     return torch.tensor(0)
        # cls_avg = torch.mean(torch.stack(cls_sum).float())
        # print(cls_avg.shape)
        cls_avg = cls_sum / len(list(torch.unique(pred)))
        return 1 - cls_avg  
# #%%
# import torch
# pred = torch.tensor([1,2,2,2,3])
# for i in [1,2,3]:
#     idx = torch.where(pred == i)
#     print(idx)
# #%%

# class OrthogonalLoss(nn.Module):
#     r"""
#     Orthogonal Loss
#     """
#     def __init__(self, batch_num, dim):
#         super(OrthogonalLoss, self).__init__()
#         self.diag_tensor = torch.eye(batch_num, dim)
#         self.batch_num = batch_num

#     def forward(self, common, specific):
#         cps = torch.stack([torch.matmul(common, specific.T) for _ in range(self.batch_num)])
#         orthn_loss = torch.mean((cps - self.diag_tensor)**2)
#         return orthn_loss
class OrthogonalLoss(nn.Module):
    def __init__(self):
        super(OrthogonalLoss, self).__init__()
        # self.batch_num = batch_num
        # self.dim = di
       # Compute the dot product between a and b
    def forward(self,common,specific):
        # common.require_grad = True
        # specific.require_grad = True

        loss = []
        for i in range(common.shape[0]):
            a = common[i]
            b = specific[i]
            dot_product = torch.dot(a, b)
            loss.append(torch.mean(torch.abs(dot_product)))
        loss= (torch.stack(loss, dim=0)).squeeze()
            # loss = torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)
        return torch.mean(loss)

    # Compute the loss as the absolute value of the dot product
    # loss = 
    
    # return loss


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
def cosine_pairwise_loss():
    loss = CosinePairwiseLoss()
    return loss 

@LOSS_REGISTRY.register()
def cosine_loss(logit, target):
    loss = CosineLoss(logit.shape[1], target.max() + 1)(logit, target)
    return loss

@LOSS_REGISTRY.register()
def orthogonal_loss():
    loss = OrthogonalLoss()
    return loss

@LOSS_REGISTRY.register()
def orthogonal_projection_loss(logit):
    loss = OrthogonalProjectionLoss()(logit)
    return loss