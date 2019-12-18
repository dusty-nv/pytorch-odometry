#
# Note -- this training script is tweaked from the original at:
#           https://github.com/pytorch/examples/tree/master/imagenet
#
# For a step-by-step guide to transfer learning with PyTorch, see:
#           https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
#
import argparse
import os
import random
import shutil
import time
import math
import warnings
import PIL.Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from models import create_model, get_model_names
from datasets import create_dataset, get_dataset_names

model_names = get_model_names()
dataset_names = get_dataset_names()

#
# parse command-line arguments
#
parser = argparse.ArgumentParser(description='PyTorch OdometryNet Training')

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', default='tum', help='dataset type: ' + ' | '.join(dataset_names) + ' (default: tum)')
parser.add_argument('--model-dir', type=str, default='', 
				help='path to desired output directory for saving model '
					'checkpoints (default: current directory)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    #choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--resolution', default=224, type=int, metavar='N',
                    help='input NxN image resolution of model (default: 224x224) '
                         'note than Inception models should use 299x299')
parser.add_argument('--width', default=0, type=int)
parser.add_argument('--height', default=0, type=int)
parser.add_argument('--input-channels', default=3, type=int, dest='input_channels')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--epochs', default=35, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 8), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-decay', '--learning-rate-decay', default=30, type=int,
                    metavar='LR-DECAY', dest='lr_decay',
                    help='the number of epochs after which to decay the learning date (default: 30 epochs)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--plot', dest='plot', action='store_true', default=True)
parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=True,
                    help='use pre-trained model')
parser.add_argument('--no-pretrained', dest='no_pretrained', action='store_true',
                    help='do not use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


best_names  = ['Avg Loss', 'Path ATE', 'Drift RPE', 'Drift %']
best_format = ['{:.4e}', '{:.4e}', '{:.4e}', '{:9.6f}%']
best_stats  = [1000000, 1000000, 1000000, 1000000]
best_epoch  = [0, 0, 0, 0]

#
# initiate worker threads (if using distributed multi-GPU)
#
def main():
    args = parser.parse_args()

    if args.width <= 0:
        args.width = args.resolution

    if args.height <= 0:
        args.height = args.resolution

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    #if args.gpu is not None:
    #    warnings.warn('You have chosen a specific GPU. This will completely '
    #                  'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


#
# worker thread (per-GPU)
#
def main_worker(gpu, ngpus_per_node, args):
    global best_stats
    global best_drift

    args.gpu = gpu

    print("=> torch:  " + str(torch.__version__))

    if args.gpu is not None:
        print("=> use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)


    # select the desired dataset
    train_dataset = create_dataset(args.dataset, root_dir=args.data, type='train', input_channels=args.input_channels, input_resolution=(args.width, args.height))
    val_dataset = create_dataset(args.dataset, root_dir=args.data, type='val', input_channels=args.input_channels, input_resolution=(args.width, args.height))
    
    output_dims = train_dataset.output_dims()

    print('=> dataset:  ' + args.dataset)
    print('=> dataset training images:   ' + str(len(train_dataset)))
    print('=> dataset validation images: ' + str(len(val_dataset)))
    print('=> dataset input channels: ' + str(train_dataset.input_channels))
    print('=> dataset input mean:     ' + str(train_dataset.input_mean))
    print('=> dataset input std_dev:  ' + str(train_dataset.input_std))
    print('=> dataset output dims:    ' + str(output_dims))

    # data transforms
    normalize = transforms.Normalize(mean=train_dataset.input_mean,
                                     std=train_dataset.input_std)

    train_dataset.transform = transforms.Compose([
            #transforms.Resize((args.height, args.width), interpolation=PIL.Image.NEAREST),	# resize now done in dataset
            #transforms.RandomResizedCrop(args.resolution),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    val_dataset.transform = transform=transforms.Compose([
            #transforms.Resize((args.height, args.width), interpolation=PIL.Image.NEAREST),	# resize now done in dataset
            #transforms.Resize(256),
            #transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalize,
        ])

    # data loaders
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # determined if using pre-trained (the default)
    if args.pretrained and not args.no_pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        pretrained = True
    else:
        print("=> creating model '{}'".format(args.arch))
        pretrained = False

    # create the model architecture
    model = create_model(args.arch, pretrained=pretrained, 
                         input_channels=args.input_channels, 
                         outputs=output_dims)

    print("=> model inputs:   {:d}x{:d}x{:d}".format(args.width, args.height, args.input_channels))
    print("=> model outputs:  {:d}".format(output_dims))

    # transfer the model to the GPU that it should be run on
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda(args.gpu) #nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # if in evaluation mode, only run validation
    if args.evaluate:
        validate(val_loader, model, criterion, args.start_epoch, output_dims, args)
        return

    # train for the specified number of epochs
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # decay the learning rate
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, output_dims, args)

        # evaluate on validation set
        val_stats = validate(val_loader, model, criterion, epoch, output_dims, args)

        # compare results to best
        is_best_stats = [a < b for a, b in zip(val_stats, best_stats)]

        print('Best Results (through epoch {:d})'.format(epoch))

        for n in range(len(best_stats)):
            if is_best_stats[n]:
                 best_stats[n] = val_stats[n]
                 best_epoch[n] = epoch

            fmt_str = ' * {:<10s} ' + best_format[n] + '  (epoch {:d})'
            print(fmt_str.format(best_names[n], best_stats[n], best_epoch[n]))

        # save model checkpoint
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'width': args.width,
                'height': args.height,
                'input_channels': args.input_channels,
                'output_dims': output_dims,
                'pretrained': pretrained,
                'state_dict': model.state_dict(),
                'best_loss': val_stats[0], #loss,
                'path_error': val_stats[1], #path_error,
                'drift_error': val_stats[2], #drift_error,
                'drift_pct': val_stats[3],
                'optimizer' : optimizer.state_dict(),
            }, is_best_stats, args)

#
# train one epoch
#
def train(train_loader, model, criterion, optimizer, epoch, output_dims, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    #top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses], #[batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    # get the start time
    epoch_start = time.time()
    end = epoch_start

    # train over each image batch from the dataset
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)
	
        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    print("Epoch: [{:d}] completed, elapsed time {:6.3f} seconds".format(epoch, time.time() - epoch_start))


#
# measure model performance across the val dataset
#
def validate(val_loader, model, criterion, epoch, output_dims, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses], #[batch_time, losses, top1],
        prefix='Test: ')

    # initialize statistics
    if args.plot:
        import matplotlib.pyplot as plt

        pose = val_loader.dataset.initial_pose()
        pose_gt = val_loader.dataset.initial_pose()

        pose_xyz = [[0.0], [0.0], [0.0]]
        pose_xyz_gt = [[0.0], [0.0], [0.0]]

    path_error    = 0.0
    drift_error   = 0.0
    drift_pct     = 0.0
    total_dist_gt = 0.0

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
           
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

            if args.evaluate:
                #print("{:04d}:  IMG={:s}".format(i, str(images)))
                print("{:04d}:  OUT^={:s}  GT^={:s}".format(i, str(output), str(target)))

            if args.plot:
                batch_size = len(output)

                for n in range(batch_size):
                    # convert the outputs back into original ranges
                    output_unnorm = val_loader.dataset.unnormalize(output.cpu().numpy()[n])
                    target_unnorm = val_loader.dataset.unnormalize(target.cpu().numpy()[n])

                    if args.evaluate:
                        print("{:04d}:  OUT:={:s}  GT:={:s}".format(i, str(output_unnorm), str(target_unnorm)))

                    # update the vel/heading pose
                    pose, xyz_delta = val_loader.dataset.pose_update(pose, output_unnorm)
                    pose_gt, xyz_delta_gt = val_loader.dataset.pose_update(pose_gt, target_unnorm)

                    # calc the position delta
                    #xyz_delta = val_loader.dataset.position_update(pose)
                    #xyz_delta_gt = val_loader.dataset.position_update(pose_gt)

                    # update the next position
                    xyz = []
                    xyz_gt = []
                    
                    prev_idx = len(pose_xyz[0]) - 1

                    for m in range(len(xyz_delta)):
                        xyz.append(pose_xyz[m][prev_idx] + xyz_delta[m])
                        xyz_gt.append(pose_xyz_gt[m][prev_idx] + xyz_delta_gt[m])

                        pose_xyz[m].append(xyz[m])
                        pose_xyz_gt[m].append(xyz_gt[m])

                    # compute the errors
                    path_error += distance(xyz_gt, xyz)
                    drift_error += distance(xyz_delta_gt, xyz_delta)
                    total_dist_gt += magnitude(xyz_delta_gt)

        # mean the statistics
        N = len(pose_xyz[0])

        drift_pct = drift_error / total_dist_gt * 100.0
        path_error *= 1.0 / float(N)
        drift_error *= 1.0 / float(N)

        # plot the path data
        if args.plot:
             y_axis = 2 if len(pose_xyz[2]) > 1 else 1
     
             plt.plot(pose_xyz[0], pose_xyz[y_axis], 'b--', label=args.arch)
             plt.plot(pose_xyz_gt[0], pose_xyz_gt[y_axis], 'r--', label='groundtruth') #, t, t**2, 'bs', t, t**3, 'g^')
             plt.suptitle('epoch {:d} (loss={:.4e}, N={:d})'.format(epoch, losses.avg, len(val_loader.dataset)))
             plt.title('path_ATE={:.4e}, drift_RPE={:4e}, drift={:f}%'.format(path_error, drift_error, drift_pct))
             plt.legend(loc="upper left")

             # save the plot to disk
             plt_dir = ""

             if args.model_dir:
                 plt_dir = os.path.join(args.model_dir, 'plots')
                 if not os.path.exists(plt_dir):
                    os.makedirs(plt_dir)
             elif args.resume:
                 plt_dir = os.path.dirname(args.resume)

             plt.savefig(os.path.join(plt_dir, 'epoch_{:d}.jpg'.format(epoch)))
             plt.clf()

        # create list of statistics to return
        val_stats = [losses.avg, path_error, drift_error, drift_pct]

        # print out the formatted results
        print('Test Results (epoch {:d})'.format(epoch))

        for n in range(len(best_stats)):
            fmt_str = ' * {:<10s} ' + best_format[n]
            print(fmt_str.format(best_names[n], val_stats[n]))

    return val_stats


#
# vector distance/magnitude
#
def distance(a, b):
    d = 0.0
    for n in range(len(a)):
         d += math.pow(b[n] - a[n], 2)
    return math.sqrt(d)

def magnitude(a):
    d = 0.0
    for n in range(len(a)):
         d += math.pow(a[n], 2)
    return math.sqrt(d)


#
# save model checkpoint
#
def save_checkpoint(state, is_best_stats, args, filename='checkpoint.pth.tar', best_filename='model_best_{:s}.pth.tar'):

    # if saving to an output directory, make sure it exists
    if args.model_dir:
        model_path = os.path.expanduser(args.model_dir)

        if not os.path.exists(model_path):
            os.mkdir(model_path)

        filename = os.path.join(model_path, filename)
        best_filename = os.path.join(model_path, best_filename)
        #best_path_filename = os.path.join(model_path, best_path_filename)
        #best_drift_filename = os.path.join(model_path, best_drift_filename)

    # save the checkpoint
    torch.save(state, filename)

    # earmark the best checkpoint
    has_best = False

    for n in range(len(is_best_stats)):
        if is_best_stats[n]:
            stat_name_mod = best_names[n].lower().replace(" ", "_").replace("%", "pct")
            copy_filename = best_filename.format(stat_name_mod)
            shutil.copyfile(filename, copy_filename)
            print("saved best {:s} model to: ".format(best_names[n]).ljust(32) + copy_filename)
            has_best = True

    #if is_best_loss:
    #    shutil.copyfile(filename, best_filename)
    #    print("saved best loss model to:  " + best_filename)

    #if is_best_path:
    #    shutil.copyfile(filename, best_path_filename)
    #    print("saved best path model to:  " + best_path_filename)

    #if is_best_drift:
    #    shutil.copyfile(filename, best_drift_filename)
    #    print("saved best drift model to: " + best_drift_filename)

    if not has_best:
        print("saved checkpoint to:  " + filename)



#
# statistic averaging
#
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


#
# progress metering
#
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


#
# learning rate decay
#
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.lr_decay)) #30))
    print('learning_rate => {:f}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#
# compute the accuracy for a given result
#
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
