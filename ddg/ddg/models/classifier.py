#%%
import torch
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import numpy as np

x = np.random.randn(256, 7)
x = torch.tensor(x, dtype=torch.float32)
si = pairwise_cosine_similarity(x)
print(si.shape)
#%%
tmp = torch.triu(si)
print(tmp)
print(tmp.shape) 
#%%
tmp_sum = torch.sum(tmp)
#%%
tmp = tmp / tmp.shape[0]
print(tmp)
#%%
tmp.shape
#%%
import torch
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

from ddg.utils import load_state_dict
from ddg.utils import MODELS_REGISTRY
from .conv import Conv2dDynamic
from .mixstyle import MixStyle
from athena.kas import KernelActiveSubspaces

__all__ = ['csd', 'fc']


    

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
        self.specific_term = torch.einsum('kd, d', self.specific_weight, self.domain_weight)
     

        self.specific_softmax_weight = torch.einsum('k, kfc -> fc', self.specific_weight, self.softmax_weight)
        self.specific_softmax_bias = torch.einsum('k, kfc -> fc', self.specific_weight, self.softmax_bias)
        self.specific_logit = torch.einsum('bf, fc -> bc', x, self.specific_softmax_weight) \
                             + (torch.one([x.shape[0], 1] * self.specific_softmax_bias))
        
        return self.common_logit, self.specific_logit


class FC(nn.Module):
    def __init__(self, in_features, num_classes):
        super(FC, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, features):
        logit = self.fc(features)   
        return logit


@MODELS_REGISTRY.register()
def csd(name, args, from_name=None) -> CSD:
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = CSD(args.dim_features, args.num_classes, args.num_domains, rank=2)

    return model

@MODELS_REGISTRY.register()
def fc(name, args, from_name=None) -> FC:
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FC(args.dim_features, args.num_classes)

    return model




