
from PIL import Image

import numpy as np

import torch.utils.data
import torchvision.transforms as transforms


#
# load image
#
def load_image(path, type='L', resolution=(-1, -1)):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		img = img.convert(type)

		if resolution[0] > 0 and resolution[1] > 0:
			img = img.resize( resolution, Image.NEAREST )

		return img

#
# calculate dataset statistics
#
def calc_dataset_stats(dataset, batch_size=128, workers=8):
	# cache these settings to restore later
	transform = dataset.transform
	scale_outputs = dataset.scale_outputs
	normalize_outputs = dataset.normalize_outputs

	dataset.transform = transforms.ToTensor()
	dataset.scale_outputs = False
	dataset.normalize_outputs = False

	# create a dataloader
	dataloader = torch.utils.data.DataLoader(
        			dataset, batch_size=batch_size, shuffle=False,
        			num_workers=workers)

	# compute output range
	output_dims  = dataset.output_dims()
	output_range = [[1000000.0, -1000000.0] for n in range(output_dims)]

	for i, (images, target) in enumerate(dataloader, 0):
		for n in range(len(target)):
			for m in range(output_dims):
				x = float(target[n][m])

				output_range[m][0] = min(x, output_range[m][0])
				output_range[m][1] = max(x, output_range[m][1])

	print(' ')
	print('OUTPUT RANGE')
	print(output_range)
			
	dataset.output_range  = output_range
	dataset.scale_outputs = True

	# compute mean/std-dev
	input_mean = []
	input_std0 = []
	#input_std1 = []

	output_mean = []
	output_std0 = []
	#output_std1 = []

	for i, (images, target) in enumerate(dataloader, 0):
		# shape (batch_size, 3, height, width)
		numpy_image = images.numpy()
		numpy_target = target.numpy()

		# shape (3,)
		batch_input_mean = np.mean(numpy_image, axis=(0,2,3))
		batch_input_std0 = np.std(numpy_image, axis=(0,2,3))
		#batch_input_std1 = np.std(numpy_image, axis=(0,2,3), ddof=1)

		input_mean.append(batch_input_mean)
		input_std0.append(batch_input_std0)
		#input_std1.append(batch_input_std1)

		batch_output_mean = np.mean(numpy_target, axis=(0))
		batch_output_std0 = np.std(numpy_target, axis=(0))
		#batch_output_std1 = np.std(numpy_target, axis=(0), ddof=1)

		output_mean.append(batch_output_mean)
		output_std0.append(batch_output_std0)
		#output_std1.append(batch_output_std1)

	# shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
	input_mean = np.array(input_mean).mean(axis=0).tolist()
	input_std0 = np.array(input_std0).mean(axis=0).tolist()
	#input_std1 = np.array(input_std1).mean(axis=0)

	print(' ')
	print('INPUT')
	print('=> mean:       ' + str(input_mean))
	print('=> std_dev_0:  ' + str(input_std0))
	#print('=> std_dev_1:  ' + str(input_std1))

	output_mean = np.array(output_mean).mean(axis=0).tolist()
	output_std0 = np.array(output_std0).mean(axis=0).tolist()
	#output_std1 = np.array(output_std1).mean(axis=0)

	print(' ')
	print('OUTPUT')
	print('=> mean:       ' + str(output_mean))
	print('=> std_dev_0:  ' + str(output_std0))
	print(' ')
	#print('=> std_dev_1:  ' + str(output_std1))

	# store statistics
	dataset.input_mean = input_mean
	dataset.input_std = input_std0

	dataset.output_mean = output_mean
	dataset.output_std = output_std0

	# restore settings
	dataset.transform = transform
	dataset.scale_outputs = scale_outputs
	dataset.normalize_outputs = normalize_outputs

	del dataloader

#
# save dataset statistics
#
def save_dataset_stats(dataset, filename):
	with open(filename,"w+") as f:
		f.write("# dataset name:  {:s}\n".format(type(dataset).__name__))
		f.write("# dataset type:  {:s}\n".format(dataset.type))
		f.write("# dataset root:  {:s}\n".format(dataset.root_dir))
		f.write("# input channels\n")
		f.write("{:d}\n".format(dataset.input_channels))
		f.write("# input mean\n")
		f.write("{:s}\n".format(str(dataset.input_mean)))
		f.write("# input std_dev\n")
		f.write("{:s}\n".format(str(dataset.input_std)))
		f.write("# output range\n")
		f.write("{:s}\n".format(str(dataset.output_range)))
		f.write("# output mean\n")
		f.write("{:s}\n".format(str(dataset.output_mean)))
		f.write("# output std_dev\n")
		f.write("{:s}\n".format(str(dataset.output_std)))

#
# vector operations
#
def normalize_std(value, mean, std):
	v = []
	N = len(value)

	for n in range(N):
		v.append((value[n] - mean[n]) / std[n])

	return v

def unnormalize_std(value, mean, std):
	v = []
	N = len(value)

	for n in range(N):
		v.append((value[n] * std[n]) + mean[n])

	return v

def scale(value, value_range):
	v = []
	N = len(value)

	for n in range(N):
		v.append((value[n] - value_range[n][0]) / (value_range[n][1] - value_range[n][0]))

	return v

def unscale(value, value_range):
	v = []
	N = len(value)

	for n in range(N):
		v.append((value[n] * (value_range[n][1] - value_range[n][0])) + value_range[n][0])

	return v

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

def vector_add(a, b):
	v = []
	for n in range(len(a)):
		v.append(a[n] + b[n])
	return v

def vector_sub(a, b):
	v = []
	for n in range(len(a)):
		v.append(a[n] - b[n])
	return v

def vector_mul(a, b):
	v = []
	if isinstance(b, collections.Sequence):
		for n in range(len(a)):
			v.append(a[n] * b[n])
	else:
		for n in range(len(a)):
			v.append(a[n] * b)
	return v

def vector_cross(a, b):
	c = [a[1]*b[2] - a[2]*b[1],
          a[2]*b[0] - a[0]*b[2],
          a[0]*b[1] - a[1]*b[0]]

	return c

