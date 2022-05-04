import multiprocessing
import time
import os
import datetime
import sys
import torch


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


import transforms
from my_dataset_coco import CocoDetection
from my_dataset_voc import VOCInstances
from backbone import resnet_fpn_backbone, resnet_backbone, mobilenet_fpn_backbone
from network_files import MaskRCNN
from network_files import MultiScaleRoIPool
import train_utils.train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups, init_distributed_mode, save_on_master, mkdir

def create_model(num_classes, load_pretrain_weights=True, backbone_model='resnet50_fpn', trainable_layers=2, roi_pool_method="roi_align"):


    # resnet50 imagenet weights url: https://download.pytorch.org/models/resnet50-0676ba61.pth
    # backbone = resnet50_fpn_backbone(pretrain_path="resnet50.pth", trainable_layers=1)
    # # resnet50-c4

    # backbone = mobilenet_fpn_backbone(backbone_model_name='mobilenet_v2')
    trainable_layers = int(trainable_layers)
    if backbone_model == "resnet50_fpn":
        backbone = resnet_fpn_backbone(pretrain_path="resnet50-0676ba61.pth", trainable_layers=trainable_layers, resnet_layers=50)
    elif backbone_model == "resnet101_fpn":
        backbone = resnet_fpn_backbone(pretrain_path="resnet101-63fe2227.pth", trainable_layers=trainable_layers, resnet_layers=101)
    elif backbone_model == 'resnet50_c4':
        backbone = resnet_backbone(pretrain_path="resnet50-0676ba61.pth", trainable_layers=trainable_layers, resnet_layers=50)
    elif backbone_model == 'resnet101_c4':
        backbone = resnet_backbone(pretrain_path="resnet101-63fe2227.pth", trainable_layers=trainable_layers, resnet_layers=101)
    elif backbone_model == 'mobilenet_fpn':
        backbone = mobilenet_fpn_backbone()
    else:
        raise ValueError(f"backbone model {backbone_model} not supported yet.")

    # backbone = resnet_backbone(pretrain_path="resnet101-63fe2227.pth", trainable_layers=0, resnet_layers=101) #"resnet101-63fe2227.pth"
    # backbone = resnet_fpn_backbone(pretrain_path="resnet101-63fe2227.pth", trainable_layers=1, resnet_layers=101) #"resnet101-63fe2227.pth"

    if roi_pool_method == 'roi_align':
        model = MaskRCNN(backbone, num_classes=num_classes)
    elif roi_pool_method == 'roi_pool':
        multi_roi_pool = MultiScaleRoIPool(featmap_names=['0', '1', '2', '3'],  # 在哪些特征层进行roi pooling
                                           output_size=[7, 7])

        mask_roi_pool = MultiScaleRoIPool(featmap_names=["0", "1", "2", "3"], output_size=14)
        model = MaskRCNN(backbone, num_classes=num_classes, box_roi_pool=multi_roi_pool, mask_roi_pool=mask_roi_pool)
    else:
        raise ValueError("roi method choose from: ['roi_align','roi_pool']")

    if load_pretrain_weights:
        # coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        weights_dict = torch.load("./maskrcnn_resnet50_fpn_coco.pth", map_location="cpu")
        for k in list(weights_dict.keys()):
            if ("box_predictor" in k) or ("mask_fcn_logits" in k):
                del weights_dict[k]

        print(model.load_state_dict(weights_dict, strict=False))

    return model

from pathlib import Path
def main(args):
    init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    # output folder for current experiment
    expt_output_dir = Path('output') / args.expt_name
    expt_output_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    det_results_file = expt_output_dir / f"det_results{now}.txt"
    seg_results_file = expt_output_dir / f"seg_results{now}.txt"
    mAP_results_file = expt_output_dir / f"mAP_results{now}.txt"

    # Data loading code
    print("Loading data")

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    COCO_root = args.data_path
    data_root = args.data_path

    # load train data set
    # coco2017 -> annotations -> instances_train2017.json
    # train_dataset = CocoDetection(COCO_root, "train", data_transform["train"])
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    train_dataset = VOCInstances(data_root, year="2012", txt_name="train.txt", transforms=data_transform["train"])

    # load validation data set
    # coco2017 -> annotations -> instances_val2017.json
    # val_dataset = CocoDetection(COCO_root, "val", data_transform["val"])
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_dataset = VOCInstances(data_root, year="2012", txt_name="val.txt", transforms=data_transform["val"])

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    if args.aspect_ratio_group_factor >= 0:
        # group images by aspect ratio to save memory
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)


    data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_sampler=train_batch_sampler,
                                                    num_workers=args.workers,
                                                    collate_fn=train_dataset.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        val_dataset, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=train_dataset.collate_fn)

    print("Creating model")
    # create model num_classes equal background + classes
    model = create_model(num_classes=args.num_classes + 1,
                         load_pretrain_weights=args.pretrain,
                         backbone_model = args.backbone_model,
                         trainable_layers=args.train_layers,
                         roi_pool_method=args.roi_pool_method)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # resume from a saved model, weights, optimizer
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        utils.evaluate(model, data_loader_test, device=device)
        return

    train_loss = []
    learning_rate = []
    val_map = []

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr = utils.train_one_epoch(model, optimizer, data_loader,
                                              device, epoch, args.print_freq,
                                              warmup=True, scaler=scaler)

        # update learning rate
        lr_scheduler.step()

        # evaluate after every epoch
        if epoch > 0 and epoch % 5 == 0:
            try:
            # if epoch>0 and epoch % 5 == 0:
                det_info, seg_info = utils.evaluate(model, data_loader_test, device=device)

                if args.rank in [-1, 0]:
                    train_loss.append(mean_loss.item())
                    learning_rate.append(lr)
                    val_map.append(det_info[1])  # pascal mAP

                    # write into txt
                    with open(det_results_file, "a") as f:
                        result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                        # txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                        txt = f"{time.time()}, {epoch}, {'  '.join(result_info)}"
                        f.write(txt + "\n")

                    with open(seg_results_file, "a") as f:
                        result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
                        # txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                        txt = f"{time.time()}, {epoch}, {'  '.join(result_info)}"
                        f.write(txt + "\n")

            except:
                print(f"unable to evaluate epoch {epoch}. Continue.")

            print(f"start saving model file, epoch {epoch}")
            save_files = {'model': model_without_ddp.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'args': args,
                          'epoch': epoch}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            save_on_master(save_files,
                           os.path.join(expt_output_dir, f'model_{epoch}.pth'))


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.rank in [-1, 0]:
        # plot loss and lr curve
        if len(train_loss) != 0 and len(learning_rate) != 0:
            from plot_curve import plot_loss_and_lr
            plot_loss_and_lr(train_loss, learning_rate,output_dir=expt_output_dir)

        # plot mAP curve
        if len(val_map) != 0:
            from plot_curve import plot_map
            plot_map(val_map,output_dir=expt_output_dir)
        sys.exit("Exit plotting")


if __name__ == "__main__":
    import argparse
    torch.multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--backbone-model', default='resnet50_fpn', help='backbone model name')
    parser.add_argument('--train-layers', default=2, help='trainable layers')
    parser.add_argument('--expt-name', default='resnet50_fpn_data0_5_trainlayer-2', help='experiment name')
    parser.add_argument('--roi-pool-method',default='roi_align', help="roi method choose from: ['roi_align','roi_pool']")
    parser.add_argument('--data-path', default='./data/VOCdevkit', help="dataset")
    # parser.add_argument('--data-path', default='/data/coco2017', help='dataset')

    parser.add_argument('--epochs', default=51, type=int, metavar='N',
                        help='number of total epochs to run')
    # resume from saved model
    parser.add_argument('--resume', default='', help='resume from checkpoint')  #./output/resnet101_c4_t5/model_29.pth
    # if resume, specify which epoch to continue with.
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')

    parser.add_argument('--device', default='cuda', help='device')

    parser.add_argument('--num-classes', default=20, type=int, help='num_classes')
    # parser.add_argument('--num-classes', default=90, type=int, help='num_classes')


    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')


    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--lr', default=0.04, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')

    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--lr-step-size', default=20, type=int, help='decrease lr every step-size epochs')

    parser.add_argument('--lr-steps', default=[], nargs='+', type=int,
                        help='decrease lr every step-size epochs')  #[16, 22]

    parser.add_argument('--lr-gamma', default=0.8, type=float, help='decrease lr by a factor of lr-gamma')

    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')

    parser.add_argument('--output-dir', default='./multi_train', help='path where to save')

    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--test-only', action="store_true", help="test only")


    parser.add_argument('--world-size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", type=bool, default=False)
    parser.add_argument("--pretrain", type=bool, default=False, help="load COCO pretrain weights.")

    parser.add_argument("--amp", default=True, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()


    if args.output_dir:
        mkdir(args.output_dir)

    main(args)
    exit("exit main 1")
    sys.exit("exit main")

    """
    conda activate dl-env
    torchrun --nproc_per_node=4 train_multi_GPU.py --batch-size 4 --lr 0.01 --pretrain False --amp True

    """
