#/bin/bash
set -e

cuda="1"
folder="2"
if [[ $# == 2 ]]; then
  cuda=$1
  folder=$2
  
fi

CUDA_VISIBLE_DEVICES=$cuda python ddg/trainer/domain_mix.py --dataset PACS --transform TransformForPACS --source-domains             Cartoon Photo Sketch --target-domains ArtPainting --models backbone=resnet34_dynamic fc=fc --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/PACS_resnet34_dynamic_fc/ArtPainting/$folder
CUDA_VISIBLE_DEVICES=$cuda python ddg/trainer/domain_mix.py --dataset PACS --transform TransformForPACS --source-domains ArtPainting         Photo Sketch --target-domains Cartoon     --models backbone=resnet34_dynamic fc=fc --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/PACS_resnet34_dynamic_fc/Cartoon/$folder 
CUDA_VISIBLE_DEVICES=$cuda python ddg/trainer/domain_mix.py --dataset PACS --transform TransformForPACS --source-domains ArtPainting Cartoon       Sketch --target-domains Photo       --models backbone=resnet34_dynamic fc=fc --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/PACS_resnet34_dynamic_fc/Photo/$folder
CUDA_VISIBLE_DEVICES=$cuda python ddg/trainer/domain_mix.py --dataset PACS --transform TransformForPACS --source-domains ArtPainting Cartoon Photo        --target-domains Sketch      --models backbone=resnet34_dynamic fc=fc --epochs 50 --batch-size 64 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir ./result/PACS_resnet34__dynamic_fc/Sketch/$folder 
