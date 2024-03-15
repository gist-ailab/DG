#/bin/bash
set -e

cuda='4,6'
timestamp=`date +%Y%m%d_%H%M_%S`

dataset="PACS"
aug="None"
backbone="resnet18_dynamic"
fc="_fc"
loss="ce"


root_dir="/ailab_mat/personal/choi_sowon/result"
dir=$dataset"_"$aug"_"$backbone"_"$fc"_"$loss
exec_file=$0

if [[ $# == 10 ]]; then
  cuda=$1
  timestamp=$2
  dataset=$3
  aug=$4
  backbone=$5
  fc=$6
  loss=$7
  root_dir=$8
  dir=$9
  exec_file=$10
  
fi
# CUDA_VISIBLE_DEVICES=$cuda python ddg/trainer/main_.py --dataset $dataset --transform TransformForPACS --source-domains ArtPainting         Photo Sketch --target-domains Cartoon     --models backbone=$backbone fc=$fc --aug $aug --loss $loss --epochs 50 --batch-size 256 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir $root_dir/$dir/Cartoon/$timestamp --exec_file $exec_file

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --master_port 2244 ddg/trainer/main_dynamic.py --local_rank 4 --gpu_ids 1,2 --distributed --dataset $dataset --transform TransformForPACS --source-domains             Cartoon Photo Sketch --target-domains ArtPainting --models backbone=$backbone fc=$fc --aug $aug --loss $loss --epochs 50 --batch-size 256 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir $root_dir/$dir/ArtPainting/$timestamp --exec_file $exec_file
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=4 --master_port 2222 ddg/trainer/main_dynamic.py --local_rank 2 --gpu_ids 0,1 --distributed --dataset $dataset --transform TransformForPACS --source-domains ArtPainting         Photo Sketch --target-domains Cartoon     --models backbone=$backbone fc=$fc --aug $aug --loss $loss --epochs 50 --batch-size 256 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir $root_dir/$dir/Cartoon/$timestamp --exec_file $exec_file
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=4 --master_port 2222 ddg/trainer/main_dynamic.py --local_rank 2 --gpu_ids 0,1 --distributed --dataset $dataset --transform TransformForPACS --source-domains ArtPainting Cartoon       Sketch --target-domains Photo       --models backbone=$backbone fc=$fc --aug $aug --loss $loss --epochs 50 --batch-size 256 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir $root_dir/$dir/Photo/$timestamp --exec_file $exec_file
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=4 --master_port 2222 ddg/trainer/main_dynamic.py --local_rank 2 --gpu_ids 1,0 --distributed --dataset $dataset --transform TransformForPACS --source-domains ArtPainting Cartoon Photo        --target-domains Sketch      --models backbone=$backbone fc=$fc --aug $aug --loss $loss --epochs 50 --batch-size 256 --optimizer name=SGD lr=0.001 momentum=0.9 weight_decay=5e-4 --scheduler name=CosineAnnealingLR --pretrained --save-dir $root_dir/$dir/Sketch/$timestamp --exec_file $exec_file
