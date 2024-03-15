#/bin/bash
set -e

cuda="1"
folder="1"
if [[ $# == 2 ]]; then
  cuda=$1
  folder=$2
  timestamp='date+%Y%m%d_%H%M'
  dir="/ailab_mat/personal/choi_sowon/result/"
  
fi

CUDA_VISIBLE_DEVICES=$cuda python ddg/trainer/domain_mix.py --dataset PACS --transform TransformForPACS --source-domains             Cartoon Photo Sketch --target-domains ArtPainting --models backbone=resnet50_dynamic fc=fc --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir $dir/PACS_resnet50_dynamic_fc/ArtPainting/$timestamp
CUDA_VISIBLE_DEVICES=$cuda python ddg/trainer/domain_mix.py --dataset PACS --transform TransformForPACS --source-domains ArtPainting         Photo Sketch --target-domains Cartoon     --models backbone=resnet50_dynamic fc=fc --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir $dir/PACS_resnet50_dynamic_fc/Cartoon/$timestamp 
CUDA_VISIBLE_DEVICES=$cuda python ddg/trainer/domain_mix.py --dataset PACS --transform TransformForPACS --source-domains ArtPainting Cartoon       Sketch --target-domains Photo       --models backbone=resnet50_dynamic fc=fc --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir $dir/PACS_resnet50_dynamic_fc/Photo/$timestamp
CUDA_VISIBLE_DEVICES=$cuda python ddg/trainer/domain_mix.py --dataset PACS --transform TransformForPACS --source-domains ArtPainting Cartoon Photo        --target-domains Sketch      --models backbone=resnet50_dynamic fc=fc --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir $dir/PACS_resnet50__dynamic_fc/Sketch/$timestamp
