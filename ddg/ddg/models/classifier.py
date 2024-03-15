
import torch
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

from ddg.utils import load_state_dict
from ddg.utils import MODELS_REGISTRY
from .conv import Conv2dDynamic
from ..methods.mixstyle import MixStyle
from athena.kas import KernelActiveSubspaces
import torch
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import numpy as np

__all__ = ['csd', 'fc', '_fc']


    

class CSD(nn.Module):
    def __init__(self, in_features, num_classes, num_domains, rank=2):
        super(CSD, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_domains = num_domains
        
        self.rank = rank

        self.softmax_weight = torch.nn.Parameter(torch.normal(0, 1e-1,size=[self.rank, self.in_features, self.num_classes], dtype=torch.float, device='cuda'), requires_grad=True)
        self.softmax_bias = torch.nn.Parameter(torch.normal(0, 1e-1,size=[self.rank, self.num_classes], dtype=torch.float, device='cuda'), requires_grad=True)
        
        self.common_weight = torch.nn.Parameter(torch.normal(0, 1e-1,size=[self.rank], dtype=torch.float, device='cuda'), requires_grad=True)
        self.specific_weight = torch.nn.Parameter(torch.normal(0, 1e-1,size=[self.rank, self.domain_weight], dtype=torch.float, device='cuda'), requires_grad=True)
        
        self.embedding_matrix = torch.nn.Parameter(torch.normal(0, 1e-1,size=[self.num_domains, self.rank], dtype=torch.float, device='cuda'), requires_grad=True)        
        self.domain_weight = torch.nn.Parameter(torch.normal(0, 1e-1,size=[self.rank, self.num_domains], dtype=torch.float, device='cuda'), requires_grad=True)
    

        # weights  -> sms * weights -> sms_weights * features [= logit]
    def forward(self, x, batch_domain_idx):
        
        
        self.common_softmax_weight = torch.einsum('k, kfc -> fc', self.common_weight, self.specific_weight)
        self.common_softmax_bias = torch.einsum('k, kc -> c' , self.common_weight, self.specific_bias)
        self.common_logit = torch.einsum('bf, fc -> bc', x, self.common_softmax_weight) \
                            + (torch.one([x.shape[0], 1] * self.common_softmax_bias))
        
       
        # batch_domain_idx = [B, K]        [D, K]   [B,]
        #     [B, K]                [D, K]           
        self.domain_weight = self.embedding_matrix[batch_domain_idx]
        self.specific_term = torch.einsum('kd, d -> k' , self.specific_weight, self.domain_weight)
     

        self.specific_softmax_weight = torch.einsum('k, kfc -> fc', self.specific_weight, self.softmax_weight)
        self.specific_softmax_bias = torch.einsum('k, kfc -> fc', self.specific_weight, self.softmax_bias)
        self.specific_logit = torch.einsum('bf, fc -> bc', x, self.specific_softmax_weight) \
                             + (torch.one([x.shape[0], 1] * self.specific_softmax_bias))
        
        return self.common_logit, self.specific_logit

# class CSDv2(nn.Module):
#     def __init__(self, in_features, num_classes, num_domains, rank=2):
#         super(CSD, self).__init__()
#         self.in_features = in_features
#         self.num_classes = num_classes
#         self.num_domains = num_domains
        
#         self.rank = rank

#         self.softmax_weight = torch.nn.Parameter(torch.normal(0, 1e-1,size=[self.rank, self.in_features, self.num_classes], dtype=torch.float, device='cuda'), requires_grad=True)
#         self.softmax_bias = torch.nn.Parameter(torch.normal(0, 1e-1,size=[self.rank, self.num_classes], dtype=torch.float, device='cuda'), requires_grad=True)
        
#         # self.common_weight = torch.nn.Parameter(torch.normal(0, 1e-1,size=[self.rank], dtype=torch.float, device='cuda'), requires_grad=True)
#         # self.specific_weight = torch.nn.Parameter(torch.normal(0, 1e-1,size=[self.rank, self.domain_weight], dtype=torch.float, device='cuda'), requires_grad=True)
#         self.embedding_matrix = torch.nn.Parameter(torch.normal(0, 1e-1,size=[self.num_domain, self.rank-1], dtype=torch.float, device='cuda'), requires_grad=True)        
#         # self.domain_weight = torch.nn.Parameter(torch.normal(0, 1e-1,size=[self.rank, self.num_domains], dtype=torch.float, device='cuda'), requires_grad=True)
    

#         # weights  -> sms * weights -> sms_weights * features [= logit]
#     def forward(self, x, batch_domain_idx):
        
#         common_weight, common_bias = self.softmax_weight[0, :, :], self.softmax_bias[0, :]
#         common_logit = torch.einsum('bf, fc -> bc', x, common_weight) + common_bias

#         domain = torch.one_hot(batch_domain_idx, self.num_domains)
#         domain_weight = domain @ self.embedding_matrix

#         specific_term_weight = torch.cat(torch.ones_like(x.shape[0], 1) * specific_weight, domain_weight, dim=1)
#         specific_term_weight = torch.tanh(specific_term_weight)

#         specific_softmax_weight = torch.einsum('k, kfc -> fc', specific_term_weight, self.softmax_weight)
#         specific_softmax_bias = torch.einsum('k, kfc -> fc', specific_term_weight, self.softmax_bias)
#         specific_logit = torch.einsum('bf, fc -> bc', x, specific_softmax_weight) + specific_softmax_bias

#         return specific_logit, common_logit

#       
# c_wts = torch.tanh(c_wts)
#         w_d, b_d = torch.einsum("bk,krl->brl", c_wts, self.sms), torch.einsum("bk,kl->bl", c_wts, self.sm_biases)
#         logits_specialized = torch.einsum("brl,br->bl", w_d, x) + b_d
#         self.common_softmax_weight = torch.einsum('k, kfc -> fc', self.common_weight, self.specific_weight)
#         self.common_softmax_bias = torch.einsum('k, kc -> c' , self.common_weight, self.specific_bias)
#         self.common_logit = torch.einsum('bf, fc -> bc', x, self.common_softmax_weight) \
#                             + (torch.one([x.shape[0], 1] * self.common_softmax_bias))
        
       
#         # batch_domain_idx = [B, K]        [D, K]   [B,]
#         #     [B, K]                [D, K]           
#         self.domain_weight = self.embedding_matrix[batch_domain_idx]
#         self.specific_term = torch.einsum('kd, d -> k' , self.specific

class FC(nn.Module):
    def __init__(self, in_features, num_classes):
        super(FC, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.fc1 = nn.Linear(in_features*2, in_features)
        self.fc2 = nn.Linear(in_features, num_classes)
    
    def forward(self,common_features):
      
        common_out = self.fc2(common_features)

        return pred_out, common_out

class _FC(nn.Module):
    def __init__(self, in_features, num_classes):
        super(_FC, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes

     
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        
        out = self.fc(x)

        return out


@MODELS_REGISTRY.register()
def csd(name, args) -> CSD:
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CSD(args.out_features, args.num_classes, args.num_domains, rank=2)

    return model

@MODELS_REGISTRY.register()
def fc(name, args) -> _FC:

    
    model = _FC(args.backbone['out_features'], args.num_class)

    return model
    
@MODELS_REGISTRY.register()
def _fc(name, args) -> _FC:
    model = _FC(args.backbone['out_features'], args.num_class)

    return model
 



