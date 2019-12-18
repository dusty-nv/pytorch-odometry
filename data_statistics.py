import os
import math
import argparse
import numpy as np

import torch
import torch.utils.data
import torchvision.transforms as transforms

from datasets import create_dataset, get_dataset_names


parser = argparse.ArgumentParser(description='Dataset statistics calculator')

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', default='tum', help='dataset type: ' + ' | '.join(get_dataset_names()) + ' (default: tum)')
parser.add_argument('--input-channels', default=3, type=int, dest='input_channels')
parser.add_argument('-b', '--batch-size', default=128, type=int)
parser.add_argument('-j', '--workers', default=8, type=int)

args = parser.parse_args()


# load the dataset
dataset = create_dataset(args.dataset, root_dir=args.data, type='train', 
				     input_channels=args.input_channels, input_resolution=(224,224), 
				     normalize_output=False, scale_output=False, 
				     transform=transforms.ToTensor())

print('=> dataset:  ' + args.dataset)
print('=> dataset images:   ' + str(len(dataset)))
print('calculating...')

dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)


# compute output range
output_dims  = dataset.output_dims()
output_range = [[1000000.0, -1000000.0] for n in range(output_dims)]

for i, (images, target) in enumerate(dataloader, 0):
	batch_size = len(target)

	for n in range(batch_size):
		for m in range(output_dims):
			x = float(target[n][m])

			output_range[m][0] = min(x, output_range[m][0])
			output_range[m][1] = max(x, output_range[m][1])

print(' ')
print('OUTPUT RANGE')
print(output_range)
			
dataset.output_range = output_range
dataset.scale_output = True


# compute mean/std-dev
input_mean = []
input_std0 = []
input_std1 = []

output_mean = []
output_std0 = []
output_std1 = []

for i, (images, target) in enumerate(dataloader, 0):
	# shape (batch_size, 3, height, width)
	numpy_image = images.numpy()
	numpy_target = target.numpy()

	# shape (3,)
	batch_input_mean = np.mean(numpy_image, axis=(0,2,3))
	batch_input_std0 = np.std(numpy_image, axis=(0,2,3))
	batch_input_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)

	input_mean.append(batch_input_mean)
	input_std0.append(batch_input_std0)
	input_std1.append(batch_input_std1)

	batch_output_mean = np.mean(numpy_target, axis=(0))
	batch_output_std0 = np.std(numpy_target, axis=(0))
	batch_output_std1 = np.std(numpy_target, axis=(0), ddof=1)

	output_mean.append(batch_output_mean)
	output_std0.append(batch_output_std0)
	output_std1.append(batch_output_std1)

# shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
input_mean = np.array(input_mean).mean(axis=0)
input_std0 = np.array(input_std0).mean(axis=0)
input_std1 = np.array(input_std1).mean(axis=0)

print(' ')
print('INPUT')
print('=> mean:       ' + str(input_mean))
print('=> std_dev_0:  ' + str(input_std0))
print('=> std_dev_1:  ' + str(input_std1))

output_mean = np.array(output_mean).mean(axis=0)
output_std0 = np.array(output_std0).mean(axis=0)
output_std1 = np.array(output_std1).mean(axis=0)

print(' ')
print('OUTPUT')
print('=> mean:       ' + str(output_mean))
print('=> std_dev_0:  ' + str(output_std0))
print('=> std_dev_1:  ' + str(output_std1))

