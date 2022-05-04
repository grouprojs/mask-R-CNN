#torchrun --nproc_per_node=4 train_multi_GPU.py --lr 0.04 --expt-name resnet101_c4_data0_5_trainlayer-2 --backbone-model resnet101_c4
#torchrun --nproc_per_node=4 train_multi_GPU.py --lr 0.04 --expt-name resnet101_fpn_data0_5_trainlayer-2-new --backbone-model resnet101_fpn
#torchrun --nproc_per_node=4 train_multi_GPU.py --lr 0.04 --expt-name resnet50_c4_data0_5_trainlayer-2 --backbone-model resnet50_c4
torchrun --nproc_per_node=4 train_multi_GPU.py --lr 0.04 --expt-name mobilenet_fpn_data0_5_trainlayer-2 --backbone-model mobilenet_fpn
#
torchrun --nproc_per_node=4 train_multi_GPU.py --lr 0.04 --expt-name resnet50_fpn_data0_5_trainlayer-3 --backbone-model resnet50_fpn --train-layers 3
torchrun --nproc_per_node=4 train_multi_GPU.py --lr 0.04 --expt-name resnet50_fpn_data0_5_trainlayer-4 --backbone-model resnet50_fpn --train-layers 4
torchrun --nproc_per_node=4 train_multi_GPU.py --lr 0.04 --expt-name resnet50_fpn_data0_5_trainlayer-5 --backbone-model resnet50_fpn --train-layers 5
torchrun --nproc_per_node=4 train_multi_GPU.py --lr 0.04 --expt-name resnet50_fpn_data0_5_trainlayer-2-roipool --backbone-model resnet50_fpn --train-layers 2 --roi-pool-method roi_pool
torchrun --nproc_per_node=4 train_multi_GPU.py --lr 0.04 --expt-name resnet50_fpn_data0_5_trainlayer-1-run2 --backbone-model resnet50_fpn --train-layers 1
#torchrun --nproc_per_node=4 train_multi_GPU.py --lr 0.04 --expt-name resnet50_fpn_data0_5_trainlayer-1-run2 --backbone-model resnet50_fpn --train-layers 1
#torchrun --nproc_per_node=4 train_multi_GPU.py --lr 0.04 --expt-name resnet50_fpn_data0_5_trainlayer-0 --backbone-model resnet50_fpn --train-layers 0



torchrun --nproc_per_node=4 train_multi_GPU.py --lr 0.04 --expt-name resnet50_fpn_data0_5_trainlayer-2-mask-hourglass --backbone-model resnet50_fpn --train-layers 2
torchrun --nproc_per_node=4 train_multi_GPU.py --lr 0.04 --expt-name resnet50_fpn_data0_5_trainlayer-2-mask-fatbelly --backbone-model resnet50_fpn --train-layers 2
