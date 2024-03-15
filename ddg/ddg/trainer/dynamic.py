import torch
from ddg.trainer import Trainer
# from ddg.trainer.lrd import LRD_Trainer
import torch.nn as nn
import time
from ddg.trainer import Trainer
from ddg.utils import DATASET_REGISTRY
from ddg.utils import SAMPLERS_REGISTRY
from ddg.methods.loss import cosine_pairwise_loss, orthogonal_loss
# from typing import override
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from ddg.utils import AverageMeter
from ddg.utils import ProgressMeter

import torch
import torch.nn as nn
import os
import time
import datetime
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
__all__ = ['DDG_Trainer']


class DDG_Trainer(Trainer):
    def __init__(self):
        # Trainer.__init__(self)
        super(DDG_Trainer, self).__init__()

        self.build_transform()
        self.build_dataset()
        self.build_data_loader()
        self.build_model()
        self.model_cuda()
        self.build_optimizer()
        self.build_scheduler()
        self.build_criterion()
        self.resume_train_state()
        self.log_args_and_env()
        self.writer_init()
        
    def build_dataset(self):
        self.datasets['train'] = DATASET_REGISTRY.get(self.args.dataset)(root=self.args.root,
                                                                    domains=set(self.args.source_domains),
                                                                    splits={'train'},
                                                                    transform=self.transform.train)
        self.datasets['val'] = DATASET_REGISTRY.get(self.args.dataset)(root=self.args.root,
                                                                  domains=set(self.args.source_domains),
                                                                  splits={'val'},
                                                                  transform=self.transform.test)
        self.datasets['test'] = DATASET_REGISTRY.get(self.args.dataset)(root=self.args.root,
                                                                   domains=set(self.args.target_domains),
                                                                   splits={'test'},
                                                                   transform=self.transform.test)
    
        self.classes = self.datasets['train'].classes
    def build_data_loader(self):
    
        if self.args.sampler is None:
            sampler = DistributedSampler(self.datasets['train']) if self.args.distributed else None
            self.args.sampler = 'DistributedSampler'
        elif self.args.sampler in SAMPLERS_REGISTRY:
            sampler = SAMPLERS_REGISTRY.get(self.args.sampler)(self.datasets['train'])
        else:
            raise NotImplementedError(f"Sampler name {self.args.sampler} is not implemented yet!")

        if self.args.distributed:
          
            self._batch_size = int(self.args.batch_size/self.args.world_size)
        else: self._batch_size = self.args.batch_size
        

        self.data_loaders['train'] = DataLoader(self.datasets['train'], batch_size=self._batch_size,
                                                shuffle=(sampler is None),
                                                num_workers=self.args.workers, pin_memory=True, sampler=sampler)
        self.data_loaders['val'] = DataLoader(self.datasets['val'], batch_size=self._batch_size, shuffle=False,
                                              num_workers=self.args.workers, pin_memory=True)
        self.data_loaders['test'] = DataLoader(self.datasets['test'], batch_size=self._batch_size, shuffle=False,
                                               num_workers=self.args.workers, pin_memory=True)
    def build_criterion(self):
        self.criterion = nn.CrossEntropyLoss()
        return self.criterion
    def after_epoch(self):
        train_losses = self.meters['losses'].avg
        train_acc1 = self.meters['top1'].avg
        train_acc5 = self.meters['top5'].avg
        self.evaluate(split=self.args.val_split)
        losses = self.meters['losses'].avg
        acc1 = self.meters['top1'].avg
        acc5 = self.meters['top5'].avg
        is_best = acc1 > self.best_acc1
        self.best_acc1 = max(acc1, self.best_acc1)
        if not self.args.distributed or (self.args.distributed and self.args.rank == 0):
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalars('learning_rate', {'lr': lr}, self.epoch + 1)
            self.writer.add_scalars('losses', {'train': train_losses, 'validation': losses}, self.epoch + 1)
            self.writer.add_scalars('acc@1', {'train': train_acc1, 'validation': acc1}, self.epoch + 1)
            self.writer.add_scalars('acc@5', {'train': train_acc5, 'validation': acc5}, self.epoch + 1)
            self.save_checkpoint(is_best=is_best)
    
    def run_epoch(self):
        progress = self.progress['train']
        end = time.time()

        for i, data in enumerate(self.data_loaders['train']):
            # images, labels = images.to(self.args.local_rank).float(), labels.to(self.args.local_rank)
            data_time = time.time() - end


  
            loss, acc1, acc5, batch_size, images_vis, target_vis = self.model_forward_backward(data)

            batch_time = time.time() - end
            self.meters_update(batch_time=batch_time,
                               data_time=data_time,
                               losses=loss,
                               top1=acc1[0],
                               top5=acc5[0],
                               batch_size=batch_size)

            # measure elapsed time
            end = time.time()


            if i % self.args.log_freq == 0:
                progress.display(i)

    def model_inference(self, images):
        out = images
        for model in self.args.models:
            out = self.models[model].float()(out)
        return out

        
    def model_forward_backward(self, batch):
        images, target, _ = self.parse_batch_train(batch)
        images, target = images.to(self.args.local_rank).float(), target.to(self.args.local_rank).long()
        # self.images, self.target = images, target
        output = self.model_inference(images.float())
        loss = self.criterion(output, target)
        self.optimizer_step(loss)
        acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
        return loss, acc1, acc5, images.size(0), images, target

    def parse_batch_train(self, batch):
        images, target, domain = batch
        # args = self.args
        # if args.low_rank is not None:
        #     images = images.cuda(device=args.low_rank, non_blocking=True)
        #     target = target.cuda(device=args.low_rank, non_blocking=True)
        #     domain = domain.cuda(device=args.low_rank, non_blocking=True)
        return images.float(), target.long(), domain
    @torch.no_grad()
    def evaluate(self, split='test'):

        def run_evaluate(loader):
            args = self.args
            progress = self.progress['evaluate']
            end = time.time()
            for i, data in enumerate(loader):
                images, target = self.parse_batch_test(data)
                images, target = images.float(), target.long()
                # measure data loading time
                data_time = time.time() - end

                # compute train output
                output = self.model_inference(images.float())

    
                loss = self.criterion(output, target)
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

                # log to tensorboard
                self.writer.add_scalar('loss/train', loss, i)
                self.writer.add_scalar('acc1/train', acc1[0], i)
                self.writer.add_scalar('acc5/train', acc5[0], i)


                if i % args.log_freq == 0:
                    progress.display(i)

            progress.display_summary()

        self.models_eval()
        self.meters_reset()
        data_loader = self.data_loaders[split]
        self.logger.info(f'Do evaluation on {split} set')
        self.progress['evaluate'] = ProgressMeter(
            len(data_loader),
            [self.meters[key] for key in self.meters],
            prefix="Evaluate: [{}]".format(self.epoch))
        run_evaluate(data_loader)
            # for model in self.args.models:
            
            #     if model == 'backbone':
            #         self.out = self.models[model].float()(images)
            #         # self.features = torch.concat((self.common_features, self.specific_features), dim=1)

            #     elif model == 'fc':
            #         self.out = self.models[model].float()(self.out)
            #     else:
            #         NotImplementedError(f"Model name {model} is not implemented yet!")

            # # features, common_feature, specific_feature,\
            # # common_fc_out, features_logit = self.args.model(images)

            # # self.pred_logit = self.pred_out.argmax(dim=1, keepdim=False)
            # # print('pred_logit:' , self.pred_logit.shape)
            # # print(f'{self.pred_out.shape, self.common_fc_out.shape}')

            # loss = self.criterion(self.out, labels)
    
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
            
            # acc1, acc5 = self.accuracy(self.out, labels, topk=(1, 5))
            
            # batch_time = time.time() - end
            # self.meters_update(batch_time=batch_time,
            #                    data_time=data_time,
            #                    losses=loss,
            #                    top1=acc1[0],
            #                    top5=acc5[0],
            #                    batch_size=self._batch_size)

            # # measure elapsed time
            # end = time.time()


            # if i % self.args.log_freq == 0:
            #     progress.display(i)
            # return loss, acc1, acc5, images.size(0)

    # @torch.no_grad()
    # def evaluate(self, split='test'):
        

    #     def run_evaluate(loader):
    #         progress = self.progress['evaluate']
    #         end = time.time()
    #         for i, (images, labels, _) in enumerate(loader):
    #             images, labels = images.to(self.args.local_rank).float(), labels.to(self.args.local_rank)
    #             data_time = time.time() - end
    #             for model in self.args.models:
                
    #                 if model == 'backbone':
    #                     self.out = self.models[model].float()(images)
    #                     # self.features = torch.concat((self.common_features, self.specific_features), dim=1)

    #                 elif model == 'fc':
    #                     self.out = self.models[model].float()(self.out)
    #                 else:
    #                     NotImplementedError(f"Model name {model} is not implemented yet!")

    #             # features, common_feature, specific_feature,\
    #             # common_fc_out, features_logit = self.args.model(images)

    #             # self.pred_logit = self.pred_out.argmax(dim=1, keepdim=False)

    #             loss = self.criterion(self.out, labels)
        
    #             acc1, acc5 = self.accuracy(self.out, labels, topk=(1, 5))
    #             batch_time = time.time() - end
            

    #             self.meters_update(data_time=data_time,
    #                             batch_time=batch_time,        
    #                                     losses=loss,
    #                                     top1=acc1[0],
    #                                     top5=acc5[0],
    #                                     batch_size=images.size(0))

    #             end = time.time()
    #             if i % self.args.log_freq == 0:
    #                 progress.display(i)
                
    #         progress.display_summary() 
    #     self.models_eval()
    #     self.meters_reset()
    #     data_loader = self.data_loaders[split]
    #     self.logger.info(f'Do evaluation on {split} set')
    #     self.progress['evaluate'] = ProgressMeter(
    #         len(data_loader),
    #         [self.meters[key] for key in self.meters],
    #         prefix="Evaluate: [{}]".format(self.epoch))
    #     run_evaluate(data_loader)

    
    def run(self):
     
        # self.build_dataset()
        # self.build_model()
        # self.build_optimizer()
        # self.build_scheduler()
        # self.build_criterion()
        # self.build_data_loader()
        # self.build_logger()
        self.writer_init()
        self.meters['batch_time'] = AverageMeter('Time', ':6.3f')
        self.meters['data_time'] = AverageMeter('Data', ':6.3f')
        self.meters['losses'] = AverageMeter('Loss', ':.4e')
        self.meters['top1'] = AverageMeter('Acc@1', ':6.2f')
        self.meters['top5'] = AverageMeter('Acc@5', ':6.2f')
        time_start = datetime.datetime.now()


        if self.args.evaluate:
            self.logger.info('Evaluate started...')
            self.evaluate()
        else:
            self.logger.info('Train started...')
            self.train()
            # self.logger.info('Evaluate started...')
            # self.evaluate()
        self.writer_close()
        elapsed = datetime.datetime.now() - time_start
        self.logger.info(f"Elapsed: {elapsed}")

    def train(self):
        self.before_train()
        for self.epoch in range(self.start_epoch, self.args.epochs):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()
            # if self.epoch % self.args.checkpoint_freq == 0:
            #     self.save_checkpoint(True)
