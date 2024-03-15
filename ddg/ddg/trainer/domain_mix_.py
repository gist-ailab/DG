import torch
from ddg.trainer import Trainer
import torch.nn as nn
from ddg.utils import DATASET_REGISTRY
from ddg.utils import SAMPLERS_REGISTRY
from ddg.trainer.domain_mix import DomainMix

__all__ = ['TrainerDG']

class TrainerDG(Trainer):
    TASK = "DG"
    def __init__(self):
        super(TrainerDG, self).__init__()
    
    def build_dataset(self):
        args = self.args
        self.datasets['train'] = DATASET_REGISTRY.get(args.dataset)(root=args.root,
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
    def build_data_loader(self):
        args = self.args
        if args.sampler is None:
            sampler = DistributedSampler(self.datasets['train']) if self.args.distributed else None
            args.sampler = 'DistributedSampler'
        elif args.sampler in SAMPLERS_REGISTRY:
            sampler = SAMPLERS_REGISTRY.get(args.sampler)(self.datasets['train'])
        else:
            raise NotImplementedError(f"Sampler name {args.sampler} is not implemented yet!")
        self.samplers['train'] = sampler
        self.data_loaders['train'] = DataLoader(self.datasets['train'], batch_size=args.batch_size,
                                                shuffle=(sampler is None),
                                                num_workers=args.workers, pin_memory=True, sampler=sampler)
        self.data_loaders['val'] = DataLoader(self.datasets['val'], batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
        self.data_loaders['test'] = DataLoader(self.datasets['test'], batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.workers, pin_memory=True)
        self.classes = self.datasets['train'].classes
    
        if self.args.aug is 'domain_mix':
            def parse_batch_train(self, batch):
                images, target, domain = super(T, self).parse_batch_train(batch)
                images, target_a, target_b, lam = self.domain_mix(images, target, domain)
                return images, target, target_a, target_b, lam
                

            def model_forward_backward(self, batch):
                images, target, label_a, label_b, lam = self.parse_batch_train(batch)
                output = self.model_inference(images)
                
                loss = lam * self.criterion(output, label_a) + (1 - lam) * self.criterion(output, label_b)
                self.optimizer_step(loss)
                acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
                return loss, acc1, acc5, images.size(0), images, target
        
            
    def run_epoch(self):
        args = self.args
        progress = self.progress['train']
        end = time.time()
        for i, data in enumerate(self.data_loaders['train']):
            data_time = time.time() - end
            loss, acc1, acc5, batch_size, images_vis, target_vis = self.model_forward_backward(data)

            batch_time = time.time() - end
            self.meters_update(batch_time=batch_time,
                               data_time=data_time,
                               losses=loss.item(),
                               top1=acc1[0],
                               top5=acc5[0],
                               batch_size=batch_size)

            end = time.time()

            if i % args.log_freq == 0:
                progress.display(i)
                self.writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(self.models, images_vis, target_vis, self.classes),
                            global_step=i * len(self.data_loaders['train']) + i)

    def model_forward_backward(self, batch):
        images, target, _ = self.parse_batch_train(batch)
        images_vis, target_vis = images, target
        # self.images, self.target = images, target
        output = self.model_inference(images)
        loss = self.criterion(output, target)
        self.optimizer_step(loss)
        acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
        return loss, acc1, acc5, images.size(0), images_vis, target_vis

    def parse_batch_train(self, batch):
        images, target, domain = batch
        args = self.args
        if args.gpu is not None:
            images = images.cuda(device=args.gpu, non_blocking=True)
            target = target.cuda(device=args.gpu, non_blocking=True)
            domain = domain.cuda(device=args.gpu, non_blocking=True)
        return images, target, domain


    def __init__(self):
        super(TrainerDG, self).__init__()
        self.mix_type = self.args.domain_mix_type
        self.alpha = self.args.domain_mix_alpha
        self.beta = self.args.domain_mix_beta
        self.dist_beta = torch.distributions.Beta(self.alpha, self.beta)

    def add_extra_args(self):
        super(TrainerDG, self).add_extra_args()
        parse = self.parser
        parse.add_argument('--domain-mix-type', type=str, default='crossdomain', choices={'random', 'crossdomain'},
                           help='Mix type for DomainMix.')
        parse.add_argument('--domain-mix-alpha', type=float, default=1.0, help='alpha for DomainMix.')
        parse.add_argument('--domain-mix-beta', type=float, default=1.0, help='beta for DomainMix.')

    def model_forward_backward(self, batch):
        images, target, label_a, label_b, lam = self.parse_batch_train(batch)
        images_vis, target_vis = images, target
        output = self.model_inference(images)
        
        loss = lam * self.criterion(output, label_a) + (1 - lam) * self.criterion(output, label_b)
        self.optimizer_step(loss)
        acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
        return loss, acc1, acc5, images.size(0), images_vis, target_vis

    def parse_batch_train(self, batch):
        images, target, domain = super(DomainMix, self).parse_batch_train(batch)
        images, target_a, target_b, lam = self.domain_mix(images, target, domain)
        return images, target, target_a, target_b, lam

    def domain_mix(self, x, target, domain):
        lam = (self.dist_beta.rsample((1,)) if self.alpha > 0 else torch.tensor(1)).to(x.device)

        # random shuffle
        perm = torch.randperm(x.size(0), dtype=torch.int64, device=x.device)
        if self.mix_type == 'crossdomain':
            domain_list = torch.unique(domain)
            if len(domain_list) > 1:
                for idx in domain_list:
                    cnt_a = torch.sum(domain == idx)
                    idx_b = (domain != idx).nonzero().squeeze(-1)
                    cnt_b = idx_b.shape[0]
                    perm_b = torch.ones(cnt_b).multinomial(num_samples=cnt_a, replacement=bool(cnt_a > cnt_b))
                    perm[domain == idx] = idx_b[perm_b]
        elif self.mix_type != 'random':
            raise NotImplementedError(f"Chooses {'random', 'crossdomain'}, but got {self.mix_type}.")
        mixed_x = lam * x + (1 - lam) * x[perm, :]
        target_a, target_b = target, target[perm]
        return mixed_x, target_a, target_b, lam

    def build_criterion(self):
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing).cuda(self.args.gpu)

if __name__ == '__main__':
    DomainMix().run()
