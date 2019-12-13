#
# converts a saved PyTorch model to TensorRT format
# 
import os
import math
import torch
import argparse
import PIL.Image
import tensorrt as trt
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torch2trt import TRTModule

from datasets import MichiganIndoorDataset
from datasets import TUMSlamDataset

# parse command line
parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', default='tum', help='dataset type: tum, icl, michigan (default: tum)')
parser.add_argument('--model', type=str, default='model_trt.pth', help="path to input PyTorch model (default: model_best.pth.tar)")
parser.add_argument('--plot', type=str, default='path.jpg', help="filename of the path plot")
parser.add_argument('--resolution', default=224, type=int, metavar='N')
parser.add_argument('--width', default=0, type=int)
parser.add_argument('--height', default=0, type=int)
parser.add_argument('--input-channels', default=2, type=int, dest='input_channels')
parser.add_argument('-b', '--batch-size', default=1, type=int)
parser.add_argument('-j', '--workers', default=1, type=int)

args = parser.parse_args() 

if args.width <= 0:
	args.width = args.resolution

if args.height <= 0:
	args.height = args.resolution

print(args)


# load the dataset
if args.dataset == "tum" or args.dataset == "icl":
	dataset = TUMSlamDataset(root_dir=args.data, type='train', input_channels=args.input_channels)
elif args.dataset == "michigan":
	dataset = MichiganIndoorDataset(root_dir=args.data, type='train', input_channels=args.input_channels)

normalize = transforms.Normalize(mean=dataset.input_mean,
                                std=dataset.input_std)

dataset.transform = transforms.Compose([
       transforms.Resize((args.height, args.width), interpolation=PIL.Image.NEAREST),
       #transforms.RandomResizedCrop(args.resolution),
       #transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       normalize,
   ])

dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)


# load the torch2trt model
print('loading model:  ' + args.model)

model = TRTModule()
model.load_state_dict(torch.load(args.model))




# compute the trajectory
pose = [0.0, 0.0]     # velocity, heading
pose_gt = [0.0, 0.0]

pose_x = [0.0]
pose_y = [0.0]

pose_x_gt = [0.0]
pose_y_gt = [0.0]

for i, (images, target) in enumerate(dataloader):
	images = images.cuda()
	output = model(images)

	batch_size = len(output)

	for n in range(batch_size):
		output_unnorm = dataset.unnormalize(output.cpu().numpy()[n])
		target_unnorm = dataset.unnormalize(target.numpy()[n])

		#print("{:d} out={:s} gt={:s}".format(i, str(output_unnorm), str(target_unnorm)))

		pose[0] = output_unnorm[0]
		pose[1] += output_unnorm[1]

		pose_gt[0] = target_unnorm[0]
		pose_gt[1] += target_unnorm[1]

		prev_idx = len(pose_x) - 1

		new_x = pose_x[prev_idx] + pose[0] * math.cos(pose[1])
		new_y = pose_y[prev_idx] + pose[0] * math.sin(pose[1])

		new_x_gt = pose_x_gt[prev_idx] + pose_gt[0] * math.cos(pose_gt[1])
		new_y_gt = pose_y_gt[prev_idx] + pose_gt[0] * math.sin(pose_gt[1])

		print("{:04d} pos=[{:f}, {:f}] gt=[{:f}, {:f}]".format(i, new_x, new_y, new_x_gt, new_y_gt))

		pose_x.append(new_x)
		pose_y.append(new_y)

		pose_x_gt.append(new_x_gt)
		pose_y_gt.append(new_y_gt)


# compute the path error
path_error = 0.0
N = len(pose_x)

for n in range(N):
  path_error += math.sqrt(math.pow(pose_x_gt[n] - pose_x[n], 2) + math.pow(pose_y_gt[n] - pose_y[n], 2))

print('path error = {:f}'.format(path_error))

# plot the path data
plt.plot(pose_x, pose_y, 'b--', label=args.model)
plt.plot(pose_x_gt, pose_y_gt, 'r--', label='groundtruth')
plt.suptitle('(path_error={:f})'.format(path_error))
plt.legend(loc="upper left")
plt.savefig(args.plot)

print('saved path plot to:  ' + args.plot)


