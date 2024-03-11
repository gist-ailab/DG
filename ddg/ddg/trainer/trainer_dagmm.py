import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal as Normal_PDF
from trainer_dg import TrainerDG
from sklearn.mixture import GaussianMixture as GMM
import torch
# from igmm import IGMM
from ddg.utils import MODELS_REGISTRY
import time
from models.dagmm import DAGMM
from compute import ComputeLoss
import torch.nn as nn
"""
Takes the parameters of a Gaussian Mixture Model as an input, and creates
an object where we can evaluate the GMM's conditional distributions.

To do:

1. So far only outputs p(x1 | x2), but implementing p(x2 | x1) should be
trivial

2. Only tested on 2D Gaussian Mixture Models so far, but SHOULD be OK on
higher dimensional problems.

3. Currently only takes scale values of x1 into 'pdf_x1_cond_x2'. Should be
easy to generalise though.

P.L.Green
"""

__all__ = ['TrainerDAGMM']
# pacs = 7
class TrainerDAGMM(TrainerDG):
    def __init__(self):
        super(TrainerDAGMM, self).__init__()
        
        
    
        
    
    def build_model(self):
        """
        args.num_classes already set when loading data, you can use it for free. All other parameters that
        need to interact between models need to be set in args by yourself. For example, if model_2 use
        model_1`s output as input, we can set args.model_1_output_dim=N after init model_1, then when init
        model_2, we can let input_size=args.model_1_output_dim.

        """
        args = self.args
        from_name = None
        self.num_class = {
            'PACS' : 7,
            'OfficeHome' : 65,
            'DomainNet' : 345
        }
        for model in args.models:
            if model != 'gmm':
                args.__dict__[model] = {}
                self.models[model] = MODELS_REGISTRY.get(args.models[model])(model, args, from_name)
                from_name = model
            else: self.models['gmm'] = DAGMM(self.num_class[self.args.dataset])
        


    def add_extra_args(self):
        super(TrainerGMM, self).add_extra_args()
        parse = self.parser
        parse.add_argument('--lambda_energy', type=float, default=0.1)
        parse.add_argument('--lambda_cov', type=int, default=0.005)
        
    def models_train(self):
        for model in self.models:
               
                self.models[model].train()
                
    def model_inference(self, images):
        # For the case where the data flows linearly through all the models,
        # in other cases, you need to rewrite the function yourself.
        out = images
        for model in self.args.models:
            if model == 'gmm':
                continue
            out = self.models[model](out)
        out, gamma = self.models['gmm'](out)
        return out, gamma       
    def build_criterion(self):
        self.criterion = ComputeLoss(self.args, self.args.lambda_cov, self.args.gpu, self.num_class[self.args.dataset])
    def run_epoch(self):
        args = self.args
        progress = self.progress['train']
        end = time.time()
        for i, data in enumerate(self.data_loaders['train']):
            # measure data loading time
            data_time = time.time() - end

            loss, acc1, acc5, batch_size = self.model_forward_backward(data)

            batch_time = time.time() - end
            self.meters_update(batch_time=batch_time,
                               data_time=data_time,
                               losses=loss.item(),
                               top1=acc1[0],
                               top5=acc5[0],
                               batch_size=batch_size)

            # measure elapsed time
            end = time.time()

            if i % args.log_freq == 0:
                progress.display(i)
    @torch.no_grad()
    def evaluate(self, split='test'):

        def run_evaluate(loader):
            args = self.args
            progress = self.progress['evaluate']
            end = time.time()
            for i, data in enumerate(loader):
                images, target = self.parse_batch_test(data)

                # measure data loading time
                data_time = time.time() - end

                # compute train output
                output, gamma = self.model_inference(images)
                loss = self.criterion.forward(output, gamma, target)
                # measure train accuracy and record loss
                acc1, acc5 = self.accuracy(output, target, topk=(1, 5))

                batch_time = time.time() - end

                self.meters_update(batch_time=batch_time,
                                   data_time=data_time,
                                   losses=loss.item(),
                                   top1=acc1[0],
                                   top5=acc5[0],
                                   batch_size=images.size(0))

                # measure elapsed time
                end = time.time()

                if i % args.log_freq == 0:
                    progress.display(i)

            progress.display_summary()


    def model_forward_backward(self, batch):
        images, target, _ = self.parse_batch_train(batch)
        output, gamma = self.model_inference(images)
        loss = self.criterion.forward(output, gamma, target)
        loss.backward(retain_graph=True)
        self.optimizer_step(loss)
        acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
        return loss, acc1, acc5, images.size(0)

    def parse_batch_train(self, batch):
        images, target, domain = batch
        args = self.args
        if args.gpu is not None:
            images = images.cuda(device=args.gpu, non_blocking=True)
            target = target.cuda(device=args.gpu, non_blocking=True)
            domain = domain.cuda(device=args.gpu, non_blocking=True)
        return images, target, domain
    

if __name__ == '__main__':
    TrainerGMM().run()



