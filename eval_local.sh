DATA_DIR=/apdcephfs/share_916081/liniuslin/Collapse/datasets/imagenet100/

declare -a pathArray=("IN100_res18_moco_baseline_100epo_sym_t0d1_lr0d12_warm3_inner2048" "Fedora" "Red Hat Linux" "Ubuntu" "Debian" )

for val in "${pathArray[@]}"; do
  echo $val
  python3 evaluate_sim.py \
  --dist-url 'tcp://localhost:10003' --multiprocessing-distributed --world-size 1 --rank 0 \
  -a resnet18 \
  -b 512 \
  --lr 0.06 \
  --epochs 100 \
  --mlp --moco-t 0.1 --aug-plus --cos \
  --resume /apdcephfs/share_916081/liniuslin/Collapse/checkpoints/res18_moco_baseline_100epo_sym_lr0d12/checkpoint_0099.pth.tar \
  /apdcephfs/share_916081/liniuslin/Collapse/datasets/imagenet/ &
done
