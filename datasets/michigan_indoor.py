import os
import math
import torch
import numpy as np

from PIL import Image
from ._utils import *

from torch.utils.data import Dataset, DataLoader


class MichiganIndoorDataset(Dataset):
	"""https://deepblue.lib.umich.edu/data/concern/data_sets/3t945q88k"""

	def __init__(self, root_dir, type='train', input_channels=3, input_resolution=(224,224), 
			   normalize_outputs=True, scale_outputs=True, transform=None):
		"""
		Args:
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.type       = type		
		self.root_dir   = root_dir		
		self.transform  = transform
		self.num_images = 0
		self.images     = []
		self.poses      = []

		self.input_channels    = input_channels
		self.input_resolution  = input_resolution
		self.normalize_outputs = normalize_outputs
		self.scale_outputs 	   = scale_outputs

		self.input_mean = [0.15399212, 0.15405144]
		self.input_std  = [0.05893139, 0.05895483]

		if self.input_channels == 3:
			self.input_mean.append(0.0)
			self.input_std.append(1.0)
		elif self.input_channels == 6:
			self.input_mean = [0.19206306, 0.14614561, 0.09136458, 0.19207639, 0.14618525, 0.09142792]
			self.input_std = [0.07504185, 0.05436901, 0.04752706, 0.07504998, 0.05438552, 0.04758745]

		self.output_range = [[0.000000000, 0.112281263], [-0.047858000, 0.069044000]] #[[-1.0, 1.0] for n in range(self.output_dims())]
		self.output_mean = [0.23120594, 0.41055515]  #[0.21042515, 0.4170771] (these were with T_2 enabled)
		self.output_std = [0.07026415, 0.06002634]   #[0.05743989, 0.04943629] (these were with T_2 enabled)

		# select data subsets
		if type == 'train':
			img_subdirs = ['Dataset_+', 'Dataset_L', 'Dataset_T_1']#, 'Dataset_T_2']
		elif type == 'val':
			img_subdirs = ['Dataset_+', 'Dataset_L', 'Dataset_T_1']#, 'Dataset_T_2']

		# search for images
		for subdir in img_subdirs:
			self.search_directory(os.path.join(root_dir, 'dataset', subdir))
		
		# calc stats for train
		if type == 'train':
			calc_dataset_stats(self)

	def coordinate_space(self):
		return { 
			"x": 0,  # x coordinate
			"y": -1, # no height in Michigan
			"z": 1   # z coordinate (depth)
		}

	def output_dims(self):
		return 2   # speed, theta delta
			
	def __len__(self):
		return self.num_images

	def __getitem__(self, idx):
		img_set = 0
		img_num = 0
		img_count = 0

		# determine which subset this index lies in
		for n in range(len(self.images)):
			img_set_length = len(self.images[n]) - 1

			if idx < img_count + img_set_length:
				img_set = n
				img_num = idx - img_count
				break

			img_count += img_set_length

		#print('idx {:04d}  set {:d}  num {:03d}/{:03d}  {:s}'.format(idx, img_set, img_num, len(self.images[img_set]), self.images[img_set][img_num]))

		# load and format the images, depending on number of channels
		img_type = 'RGB' if self.input_channels == 6 else 'L'

		img_1 = load_image(self.images[img_set][img_num], type=img_type, resolution=self.input_resolution)
		img_2 = load_image(self.images[img_set][img_num+1], type=img_type, resolution=self.input_resolution)

		if self.input_channels == 2:
			# 2 channels
			#    0 R = img_1 grayscale
			#	1 G = img_2 grayscale
			img = Image.merge("LA", (img_1, img_2))
		elif self.input_channels == 3:
			# 3 channels
			#    0 R = img_1 grayscale
			#	1 G = img_2 grayscale
			#	2 B = zero 
			img = Image.merge("RGB", (img_1, img_2, Image.new('L', img_1.size, (0))))
		elif self.input_channels == 6:
			# 6 channels
			#    0 = img_1 R
			#	1 = img_1 G
			#	2 = img_1 B
			#	3 = img_2 R
			#	4 = img_2 G
			#	5 = img_2 B
			img = np.concatenate((np.asarray(img_1), np.asarray(img_2)), axis=2)
		else:
			raise Exception('invalid in_channels {:d}'.format(self.input_channels))

		# apply image transform
		if self.transform is not None:
			img = self.transform(img)
		else:
			img = torch.from_numpy(img.transpose((2, 0, 1)))

		# calc pose deltas
		pose_1 = self.poses[img_set][img_num]
		pose_2 = self.poses[img_set][img_num+1]

		pose_delta = [b - a for a, b in zip(pose_1, pose_2)]
		pose_delta = [math.sqrt(pose_delta[0] * pose_delta[0] + pose_delta[1] * pose_delta[1]), pose_delta[2]]

		# scale/normalize output
		if self.scale_outputs:
			pose_delta = scale(pose_delta, self.output_range)

		if self.normalize_outputs:
			pose_delta = normalize_std(pose_delta, self.output_mean, self.output_std)

		#print('idx {:04d}  {:s}'.format(idx, str(img)))
		return img, torch.Tensor(pose_delta)

	def unnormalize(self, value, type='output'):
		if type != 'output':
			Exception('type must be output')

		return unscale(unnormalize_std(value, self.output_mean, self.output_std), self.output_range)

	def initial_pose(self):
		return [0.0, 0.0], [0.0, 0.0]		# position, orientation

	def pose_update(self, pose, delta):
		(position, orientation) = pose

		orientation[0] = delta[0]	 # velocity
		orientation[1] += delta[1]  # heading

		dx = orientation[0] * math.cos(pose[1])
		dy = orientation[0] * math.sin(pose[1])

		# TODO if relative/absolute
		position = vector_add(position, [dx, dy])

		return position, orientation 

	def load_stats(self, dataset):
		self.input_mean   = dataset.input_mean
		self.input_std    = dataset.input_std
		self.output_range = dataset.output_range
		self.output_mean  = dataset.output_mean
		self.output_std   = dataset.output_std
		
	def save_stats(self, filename):
		save_dataset_stats(self, filename)

	def search_directory(self, dir_path):
		dir_images = []
		dir_poses = []

		# gather image files
		for n in range(1000):
			image_filename = os.path.join(dir_path, '{:04d}.ppm'.format(n))
	
			if os.path.isfile(image_filename):
				dir_images.append(image_filename)

		# parse pose file
		pose_file = open(os.path.join(dir_path, 'pose.txt'), 'r')

		while True:
			pose_line = pose_file.readline()
		
			if not pose_line:
				break
			
			pose_tokens = pose_line.rstrip().split(' ')
			
			if len(pose_tokens) != 4:
				print('invalid pose: ' + pose_line)
				break

			pose_x = float(pose_tokens[1])
			pose_y = float(pose_tokens[2])
			pose_theta = float(pose_tokens[3])

			#print('{:s}  ({:f}, {:f}) ({:f}, {:f})'.format(img_name, bbox_left, bbox_top, bbox_right, bbox_bottom))
			dir_poses.append((pose_x, pose_y, pose_theta))

		pose_file.close()

		if len(dir_images) != len(dir_poses):
			print('dataset {:s} length mismatch - {:d} images vs. {:d} poses'.format(pose_tokens[0], len(dir_images), len(dir_poses)))
			return

		print('{:s} - {:d} images'.format(dir_path, len(dir_images)))

		self.images.append(dir_images)
		self.poses.append(dir_poses)

		self.num_images += len(dir_images) - 1

