import os
import datetime
from pathlib import Path
import torch
from torchvision.ops.misc import FrozenBatchNorm2d
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import transforms
from network_files import MaskRCNN
from backbone import resnet_fpn_backbone, resnet_backbone, resnet50_fpn_backbone
from my_dataset_coco import CocoDetection
from my_dataset_voc import VOCInstances
from train_utils import train_eval_utils as utils
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups



def create_model(num_classes, load_pretrain_weights=False):
    # change from nn.BatchNorm2d to FrozenBatchNorm2d if GPU memory is small
    # FrozenBatchNorm2d cannot update its parameters
    # change batch_size to 4 or 8 if GPU memory is small
    # model is init'ed with ImageNet weights
    # to fine-tune, set trainable_layers to non-zero
    # for resnet-fpn, trainable_layers is ['layer4', 'layer3', 'layer2', 'layer1', 'conv1'], 5 means to train all
    # for resnet-c4, trainable_layer is ['layer3', 'layer2', 'layer1', 'conv1'], 4 means to trail all
    # set trainable_layers to 0 to freeze all layers without any training

    
    # Resnet-50 ImageNet weights: wget https://download.pytorch.org/models/resnet50-0676ba61.pth
    # Resnet-101 ImageNet weights: wget https://download.pytorch.org/models/resnet101-63fe2227.pth

    # # resnet50-c4
    # backbone = resnet_backbone(pretrain_path="resnet50-0676ba61.pth", trainable_layers=0, resnet_layers=50)

    # resnet101-c4
    # backbone = resnet_backbone(pretrain_path="resnet101-63fe2227.pth", trainable_layers=0, resnet_layers=101)
    # backbone = resnet_backbone(pretrain_path="resnet101.pth", trainable_layers=0, resnet_layers=101)
    backbone = resnet50_fpn_backbone(pretrain_path="resnet50.pth", trainable_layers=0)

    
    # # resnet50-fpn    
    # backbone = resnet_fpn_backbone(pretrain_path="resnet50-0676ba61.pth", trainable_layers=0, resnet_layers=50)

    # # resnet101-fpn
    # backbone = resnet_fpn_backbone(pretrain_path="resnet101-63fe2227.pth", trainable_layers=0, resnet_layers=101)
    
    # mask R-CNN
    model = MaskRCNN(backbone, num_classes=num_classes)

    if load_pretrain_weights:
        # resnet50-fpn coco weights url: "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
        weights_dict = torch.load("./maskrcnn_resnet50_fpn_coco.pth", map_location="cpu")
        for k in list(weights_dict.keys()):
            if ("box_predictor" in k) or ("mask_fcn_logits" in k):
                del weights_dict[k]

        print(model.load_state_dict(weights_dict, strict=False))

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # to save coco_info
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    det_results_file = f"det_results{now}.txt"
    seg_results_file = f"seg_results{now}.txt"

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    data_root = args.data_path
    output_dir = Path(data_root).parent/'output'

    # load train data set
    # # coco2017 -> annotations -> instances_train2017.json
    # train_dataset = CocoDetection(data_root, "train", data_transform["train"])

    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    train_dataset = VOCInstances(data_root, year="2012", txt_name="train.txt", transforms=data_transform["train"])
    train_sampler = None

    # create a batch with similar height to width ration images
    # will save GPU memory if used, so default True
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # put groups with same height to width ration into different bins
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # each batch is taken from same bin
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    # get the number of workers
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)

    if train_sampler:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    # # load validation data set
    # # coco2017 -> annotations -> instances_val2017.json
    # val_dataset = CocoDetection(data_root, "val", data_transform["val"])

    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_dataset = VOCInstances(data_root, year="2012", txt_name="val.txt", transforms=data_transform["val"])
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=nw,
                                                  collate_fn=train_dataset.collate_fn)

    # create model num_classes equal background + classes
    model = create_model(num_classes=args.num_classes + 1, load_pretrain_weights=args.pretrain)
    model.to(device)

    # print mode
    print(model)
    # from torchinfo import summary
    # summary(model, input_size=(batch_size, 3, 799, 1201))

    train_loss = []
    learning_rate = []
    val_map = []

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.lr_steps,
                                                        gamma=args.lr_gamma)
    # continue from last epoch
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # load previous saved weights
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = utils.train_one_epoch(model, optimizer, train_data_loader,
                                              device, epoch, print_freq=50,
                                              warmup=True, scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        det_info, seg_info = utils.evaluate(model, val_data_loader, device=device)

        # write detection into txt
        with open(det_results_file, "a") as f:
            # include loss and learning rate
            result_info = [f"{i:.4f}" for i in det_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        # write seg into txt
        with open(seg_results_file, "a") as f:
            # include loss and learning rate
            result_info = [f"{i:.4f}" for i in seg_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(det_info[1])  # pascal mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        torch.save(save_files, "./save_weights/model_{}.pth".format(epoch))

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # training device type
    parser.add_argument('--device', default='cuda:0', help='device')
    # training dataset root path
    parser.add_argument('--data-path', default='./data/VOCdevkit', help='dataset')
    # detection number of classes
    parser.add_argument('--num-classes', default=20, type=int, help='num_classes')
    # weights output direcctory
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # if continue to train from last epoch, specify path of last trained weight
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # if continue to train from last epich, speciy which eopch
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # total epoches
    parser.add_argument('--epochs', default=3, type=int, metavar='N',
                        help='number of total epochs to run')
    # learning rate
    parser.add_argument('--lr', default=0.004, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # momentum parameter for SGD
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # weight_decay parameter for SGD
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # for torch.optim.lr_scheduler.MultiStepLR
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    # for torch.optim.lr_scheduler.MultiStepLR
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # batch_size, set to small if GPU memory is small
    parser.add_argument('--batch_size', default=4, type=int, metavar='N', help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument("--pretrain", type=bool, default=True, help="load COCO pretrain weights.")
    # to use mixed precision training or not
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    print(args)

    # check if weights output director exists, if not, create one
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
