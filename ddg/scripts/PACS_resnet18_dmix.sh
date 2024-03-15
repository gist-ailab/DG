#/bin/bash
set -e

cuda="4"
timestamp=`date +%Y%m%d_%H%M_%S`
dir="/ailab_mat/personal/choi_sowon/result/"

if [[ $# == 3 ]]; then
  cuda=$1
  timestamp=$2
  dir=$3
  
fi

CUDA_VISIBLE_DEVICES=$cuda python ddg/trainer/domain_mix.py --dataset PACS --transform TransformForPACS --source-domains             Cartoon Photo Sketch --target-domains ArtPainting --models backbone=resnet18_dynamic fc=fc --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir $dir/PACS_resnet18_dynamic_fc_dmix/ArtPainting/$timestamp
CUDA_VISIBLE_DEVICES=$cuda python ddg/trainer/domain_mix.py --dataset PACS --transform TransformForPACS --source-domains ArtPainting         Photo Sketch --target-domains Cartoon     --models backbone=resnet18_dynamic fc=fc --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir $dir/PACS_resnet18_dynamic_fc_dmix/Cartoon/$timestamp 
CUDA_VISIBLE_DEVICES=$cuda python ddg/trainer/domain_mix.py --dataset PACS --transform TransformForPACS --source-domains ArtPainting Cartoon       Sketch --target-domains Photo       --models backbone=resnet18_dynamic fc=fc --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir $dir/PACS_resnet18_dynamic_fc_dmix/Photo/$timestamp
CUDA_VISIBLE_DEVICES=$cuda python ddg/trainer/domain_mix.py --dataset PACS --transform TransformForPACS --source-domains ArtPainting Cartoon Photo        --target-domains Sketch      --models backbone=resnet18_dynamic fc=fc --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir $dir/PACS_resnet18_dynamic_fc_dmix/Sketch/$timestamp
