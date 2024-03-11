import numpy as np
from numpy.linalg import inv
from scipy.stats import multivariate_normal as Normal_PDF
from trainer_dg import TrainerDG
from sklearn.mixture import GaussianMixture as GMM
import torch
# from igmm import IGMM
from ddg.utils import MODELS_REGISTRY
import time
import torch.nn as nn
from utils import *

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

__all__ = ['TrainerGMM']


class TrainerGMM(TrainerDG):
    def __init__(self):
        super(TrainerGMM, self).__init__()
        

    def em_gmm(self, X, gm_num):
        eps = 0.1
        max_iter = 500
        tol = 1e-5
        X = X.cpu().detach().numpy()
        # X dimensions
        n, d = X.shape
        # log-likelihood values of each iteration
        self.log_likelihood_loss = {}

        # initialization
        log_r_matrix = np.eye(gm_num)
        log_r_matrix = log_r_matrix[np.random.choice(gm_num, size=n)].T
        means = variances = pi = np.array([]) 
        iter_num = 0

            
        for it in range(max_iter):
            # M-Step
            r_k = np.sum(log_r_matrix, axis=1) + (10 * np.finfo(float).eps)  # shape: (K,), sum of elements in log_r_matrix rows
            pi = r_k / n
            means = log_r_matrix @ X / r_k[:, np.newaxis]  # (K, d)
            variances = log_r_matrix @ (X ** 2) / r_k[:, np.newaxis]
            variances -= means ** 2
            variances += eps         #avoiding zero elements for future by-zero division

            # E-Step
            sum_log_r, log_r_matrix = multivariate_gaussian(X, pi, variances, means)
            log_r_matrix = np.exp(log_r_matrix - sum_log_r)

            # compute loss
            self.log_likelihood_loss[it] = -np.sum(sum_log_r)
            loss_difference = 0
            if it > 1:
                self.loss_difference = np.abs(self.log_likelihood_loss[it] - self.log_likelihood_loss[it - 1]) / (np.abs(log_likelihood_loss[it]) + eps)
            if loss_difference <= tol:
                iter_num = it
                break     

        print("EM for GMM converged after ", iter_num + 1, "iteration, with loss: ", self.log_likelihood_loss[iter_num])
        GMM_Params = {'log_r_matrix': log_r_matrix, 'means': means, 'variances': variances, 'pi': pi}
        return GMM_Params, self.log_likelihood_loss  
    
    def build_model(self):
        """
        args.num_classes already set when loading data, you can use it for free. All other parameters that
        need to interact between models need to be set in args by yourself. For example, if model_2 use
        model_1`s output as input, we can set args.model_1_output_dim=N after init model_1, then when init
        model_2, we can let input_size=args.model_1_output_dim.

        """
        args = self.args
        from_name = None
    
        for model in args.models:
         
            args.__dict__[model] = {}
            self.models[model] = MODELS_REGISTRY.get(args.models[model])(model, args, from_name)
            from_name = model
        


        
    def models_train(self):
        for model in self.models:
                self.models[model].train()
                
    def model_inference(self, images):
     
        out = images
        for model in self.args.models:
            specific, common = self.models[model](out)
        
        sms = self.models[model].sms
        K = 2
        diag_tensor = torch.stack([torch.eye(K) for _ in range(self.num_classes)], dim=0).cuda()
        cps = torch.stack([torch.matmul(sms[:, :, _], torch.transpose(sms[:, :, _], 0, 1)) for _ in range(self.num_classes)], dim=0)
        return specific, common, diag_tensor, cps
           
    def parse_batch_train(self, batch):
        images, target, domain = batch
        print('batch ', batch)
        print('images ', images.shape)
        print('target ', target.shape)
        print('domain ', domain.shape)
        
        if self.args.gpu is not None:
            images = images.cuda(device=self.args.gpu, non_blocking=True)
            target = target.cuda(device=self.args.gpu, non_blocking=True)
            domain = domain.cuda(device=self.args.gpu, non_blocking=True)
        return images, target, domain
    
    def model_forward_backward(self, batch):
        images, target, domain = self.parse_batch_train(batch)
        
        GMM_Params = {}
        log_likelihood_loss = []
        prior = np.zeros(self.num_classes)
        gmm_loss = []
        specific, common, diag_tensor, cps = self.model_inference(images)
        
        for c in range(self.num_classes):
            print('... GMM is fitting to digit  ', c)
            c_common = common[target == c]
            GMM_Params[c], log_likelihood_loss[c] = self.em_gmm(c_common, self.num_classes)
            print('GMM parameters computed for class = ', c)
            prior[c] = (c_common.shape[0] / images.shape[0])
            gmm_loss += log_likelihood_loss[c]
            print('....GMM parameteres for each digit were computed!')


        gmm_loss = np.mean(gmm_loss)
        orth_loss = torch.mean((cps - diag_tensor)**2)
        class_loss = nn.CrossEntropyLoss()(common, target)

        loss = class_loss + orth_loss + gmm_loss     
        loss.backward(retain_graph=True)
      
        self.optimizer_step(loss)

        acc1, acc5 = self.accuracy(common, target, topk=(1, 5))
        return loss, acc1, acc5, images.size(0)
    
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
                GMM_Params = {}
                self.log_likelihood_loss = []
                prior = np.zeros(self.num_classes)
                self.gmm_loss = []
                # measure data loading time
                data_time = time.time() - end

                # compute train output

                specific, common, diag_tensor, cps = self.model_inference(images)
                for c in range(self.num_classes):
                    print('... GMM is fitting to digit  ', c)
                    c_common = common[target == c]
                    GMM_Params[c], self.log_likelihood_loss[c] = self.em_gmm(c_common, self.num_class)
                    print('GMM parameters computed for class = ', c)
                    prior[c] = (c_common.shape[0] / images.shape[0])
                    self.gmm_loss += self.log_likelihood_loss[c]

                    print('....GMM parameteres for each digit were computed!')


                # GMM_Params, self.log_likelihood_loss = self.em_gmm(common, self.num_class)
                self.gmm_loss = np.mean(self.gmm_loss)
                self.orth_loss = torch.mean((cps - diag_tensor)**2)
                self.class_loss = nn.CrossEntropyLoss()(common, target)
                loss = self.class_loss + self.orth_loss + self.gmm_loss
      

                # measure train accuracy and record loss
                acc1, acc5 = self.accuracy(common, target, topk=(1, 5))

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


    


    

if __name__ == '__main__':
    TrainerGMM().run()

    # def add_extra_args(self):
       
    #     parse = self.parser
    #     parse.add_argument('--lambda_energy', type=float, default=0.1)
    #     parse.add_argument('--lambda_cov', type=int, default=0.005)

