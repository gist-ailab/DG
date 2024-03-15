 import torch

import torch.nn as nn
from ddg.methods.loss import *
from ddg.trainer import Trainer
from ddg.utils import DATASET_REGISTRY



__all__ = ['LRD']


class LRD(Trainer):
 
    def __init__(self, args):
        super(LRD, self).__init__()
~       self.args = args


    def build_dataset(self):
        
        self.datasets['train'] = DATASET_REGISTRY.get(self.args.dataset)(root=args.root,
                                                                    domains=set(args.source_domains),
                                                                    splits={'train'},
                                                                    transform=self.transform.train)
        self.datasets['val'] = DATASET_REGISTRY.get(args.dataset)(root=args.root,
                                                                    domains=set(args.source_domains),
                                                                    splits={'val'},
                                                                    transform=self.transform.test)
        self.datasets['test'] = DATASET_REGISTRY.get(args.dataset)(root=args.root,
                                                                    domains=set(args.target_domains),
                                                                    splits={'test'},
                                                                    transform=self.transform.test)

        self.classes = self.datasets['train'].classes

 
    def build_criterion(self, logit, target, common_feature, specific_feature, fc_common_out, pred):
        self.ce = nn.CrossEntropyLoss()(logit, target)
        self.orthogonal = orthogonal_loss()(common_feature, specific_feature)
        self.cosine_pairwise = cosine_pairwise_loss()(fc_common_out, pred)
        return self.ce + self.orthogonal + self.cosine_pairwise
    
    def model_inference(self, images):
        out = images.to(self.args.local_rank).float()
    
        for model in self.args.models:
           
            if model == 'backbone':
                self.features, self.common_features, self.specific_features = self.models[model].float()(out)
                # self.features = torch.concat((self.common_features, self.specific_features), dim=1)

            elif model == 'fc':
                self.pred_out, self.common_out = self.models[model].float()(self.features, self.common_features)
            else:
                NotImplementedError(f"Model name {model} is not implemented yet!")
        return self.features, self.common_features, self.specific_features, self.common_out, self.pred_out


    def model_forward_backward(self, batch):
        images, target, _ = self.parse_batch_train(batch)
        images, target = images.to(self.args.local_rank).float(), target.to(self.args.local_rank)
        feature, common_feature, specific_feature, fc_common_out, logit = self.model_inference(images)
        pred = logit.argmax(dim=1, keepdim=False)
        
        loss = self.make_criterion(logit, target, common_feature, specific_feature, fc_common_out, pred)
        # loss = torch.tensor(loss, requires_grad=True)
        loss.backward()
        self.optimizer.step()
        acc1, acc5 = self.accuracy(logit, target, topk=(1, 5))
        return loss, acc1, acc5, images.size(0), images, target


if __name__ == '__main__':
    LRD().run()
    se











    # def parse_batch_train(self, batch):
    #     images, target, domain = super(DomainMix, self).parse_batch_train(batch)
    #     images, target_a, target_b, lam = self.domain_mix(images, target, domain)
    #     return images, target, target_a, target_b, lam

#     def domain_mix(self, x, target, domain):
#         lam = (self.dist_beta.rsample((1,)) if self.alpha > 0 else torch.tensor(1)).to(x.device)

#         # random shuffle
#         perm = torch.randperm(x.size(0), dtype=torch.int64, device=x.device)
#         if self.mix_type == 'crossdomain':
#             domain_list = torch.unique(domain)
#             if len(domain_list) > 1:
#                 for idx in domain_list:
#                     cnt_a = torch.sum(domain == idx)
#                     idx_b = (domain != idx).nonzero().squeeze(-1)
#                     cnt_b = idx_b.shape[0]
#                     perm_b = torch.ones(cnt_b).multinomial(num_samples=cnt_a, replacement=bool(cnt_a > cnt_b))
#                     perm[domain == idx] = idx_b[perm_b]
#         elif self.mix_type != 'random':
#             raise NotImplementedError(f"Chooses {'random', 'crossdomain'}, but got {self.mix_type}.")
#         mixed_x = lam * x + (1 - lam) * x[perm, :]
#         target_a, target_b = target, target[perm]
#         return mixed_x, target_a, target_b, lam

#     def build_criterion(self):
#         self.criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing).cuda(self.args.gpu)

