import os
import math
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MichiganIndoorDataset(Dataset):
	"""https://deepblue.lib.umich.edu/data/concern/data_sets/3t945q88k"""

	def __init__(self, root_dir, type='train', input_channels=3, normalize_output=True, transform=None):
		"""
		Args:
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.type = type
		self.root_dir = root_dir		
		self.transform = transform
		self.input_channels = input_channels
		self.normalize_output = normalize_output
		self.num_images = 0

		self.velocity_min = 10000.0
		self.velocity_max = -10000.0

		self.heading_min = 10000.0
		self.heading_max = -10000.0

		self.input_mean = [0.15399212, 0.15405144]
		self.input_std = [0.05893139, 0.05895483]

		if self.input_channels == 3:
			self.input_mean.append(0.0)
			self.input_std.append(1.0) 

		self.output_mean = [0.21042515, 0.4170771]
		self.output_std = [0.05743989, 0.04943629]

		self.images = []
		self.poses = []

		if type == 'train':
			img_subdirs = ['Dataset_+', 'Dataset_L', 'Dataset_T_1', 'Dataset_T_2']
		elif type == 'val':
			img_subdirs = ['Dataset_+', 'Dataset_L', 'Dataset_T_1', 'Dataset_T_2']

		for subdir in img_subdirs:
			dir_path = os.path.join(root_dir, 'dataset', subdir)

			dir_images = []
			dir_poses = []

			# gather image files
			for n in range(1000):
				image_filename = os.path.join(dir_path, '{:04d}.ppm'.format(n))
		
				if os.path.isfile(image_filename):
					dir_images.append(image_filename)

			# 
			last_x = 0.0
			last_y = 0.0
			last_theta = 0.0

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

				# determine range of velocity/heading data
				delta_x = pose_x - last_x
				delta_y = pose_y - last_y
				delta_theta = pose_theta - last_theta

				velocity = math.sqrt(delta_x * delta_x + delta_y * delta_y)

				self.velocity_max = max(velocity, self.velocity_max)
				self.velocity_min = min(velocity, self.velocity_min)

				self.heading_max = max(delta_theta, self.heading_max)
				self.heading_min = min(delta_theta, self.heading_min)

				last_x = pose_x
				last_y = pose_y
				last_theta = pose_theta
			
			pose_file.close()

			if len(dir_images) != len(dir_poses):
				print('dataset {:s} length mismatch - {:d} images vs. {:d} poses'.format(pose_tokens[0], len(dir_images), len(dir_poses)))
				continue

			print('{:s} - {:d} images'.format(dir_path, len(dir_images)))

			self.images.append(dir_images)
			self.poses.append(dir_poses)

			self.num_images += len(dir_images) - 1

		print('=> output range:')
		print('      - velocity  (min={:.9f}) (max={:.9f}) (range={:.9f})'.format(self.velocity_min, self.velocity_max, self.velocity_max - self.velocity_min))
		print('      - heading Δ (min={:.9f}) (max={:.9f}) (range={:.9f})'.format(self.heading_min, self.heading_max, self.heading_max - self.heading_min))

		#print(str(self.num_images))
		#print(self.images)
		
	def output_dims(self):
		return 2   # speed, theta delta     #3	# x, y, theta
			
	def __len__(self):
		return self.num_images

	def __getitem__(self, idx):
		img_set = 0
		img_num = 0
		img_count = 0

		for n in range(len(self.images)):
			img_set_length = len(self.images[n]) - 1

			if idx < img_count + img_set_length:
				img_set = n
				img_num = idx - img_count
				break

			img_count += img_set_length

		#print('idx {:04d}  set {:d}  num {:03d}/{:03d}  {:s}'.format(idx, img_set, img_num, len(self.images[img_set]), self.images[img_set][img_num]))

		rgb_diff = False

		if rgb_diff:
			img_1 = load_image(self.images[img_set][img_num], type='RGB', resolution=224)
			img_2 = load_image(self.images[img_set][img_num+1], type='RGB', resolution=224)
			
			img = np.asarray(img_2).astype(np.float32) - np.asarray(img_1).astype(np.float32) #Image.fromarray(np.asarray(img_2) - np.asarray(img_1))
			#print('idx {:04d}  img max diff:  {:f}'.format(idx, np.max(img)))			
			img = (img + 255.0) / 510.0 #/ 2.0
			#img = Image.fromarray(img.astype(np.uint8))		
		else:
			img_1 = load_image(self.images[img_set][img_num])
			img_2 = load_image(self.images[img_set][img_num+1])

			if self.input_channels == 2:
				img = Image.merge("LA", (img_1, img_2))
			elif self.input_channels == 3:
				gray_diff = False

				if gray_diff:
					img_0 = np.asarray(img_2).astype(np.float32) - np.asarray(img_1).astype(np.float32)
					img_0 = (img_0 + 255.0) * 0.5
					img_0 = Image.fromarray(img_0.astype(np.uint8))
				else:
					img_0 = Image.new('L', img_1.size, (0))

				img = Image.merge("RGB", (img_1, img_2, img_0))
			else:
				raise Exception('invalid in_channels {:d}'.format(self.input_channels))

		if self.transform is not None:
			img = self.transform(img)
		else:
			img = torch.from_numpy(img.transpose((2, 0, 1)))

		# calc pose deltas
		pose_1 = self.poses[img_set][img_num]
		pose_2 = self.poses[img_set][img_num+1]

		pose_delta = [b - a for a, b in zip(pose_1, pose_2)]

		#pose_x = pose_2[0] - pose_1[0]
		#pose_y = pose_2[1] - pose_1[1]
		#pose_theta = pose_2[2] - pose_1[2] 

		#print(img)
		#print('idx {:04d}  d_x {:f} d_y {:f} d_theta {:f}'.format(idx, pose_x, pose_y, pose_theta))
		#print('idx {:04d}  x_1 {:f} y_1 {:f} t_1 {:f} x_2 {:f} y_2 {:f} t_2 {:f} d_x {:f} d_y {:f} d_theta {:f}'.format(idx, pose_1[0], pose_1[1], pose_1[2], pose_2[0], pose_2[1], pose_2[2], pose_x, pose_y, pose_theta))

		pose_speed = True

		if pose_speed:
			pose_delta = [ math.sqrt(pose_delta[0] * pose_delta[0] + pose_delta[1] * pose_delta[1]), pose_delta[2] ]
			#print('idx {:04d}  velocity {:f} d_theta {:f}'.format(idx, pose_delta[0], pose_delta[1]))

		scale_output = True

		if scale_output:
			pose_delta = [ scale(pose_delta[0], self.velocity_min, self.velocity_max),
						scale(pose_delta[1], self.heading_min, self.heading_max) ]

		if self.normalize_output:
			pose_delta = normalize_std(pose_delta, self.output_mean, self.output_std)

		#print('idx {:04d}  {:s}'.format(idx, str(img)))
		return img, torch.Tensor(pose_delta)

	def unnormalize(self, value, type='output'):
		if type != 'output':
			Exception('type must be output')

		output_min = [self.velocity_min, self.heading_min]
		output_max = [self.velocity_max, self.heading_max]

		return unscale(unnormalize_std(value, self.output_mean, self.output_std), output_min, output_max)


def normalize_std(value, mean, std):
	v = []
	N = len(value)

	for n in range(N):
		v.append( (value[n] - mean[n]) / std[n] )

	return v

def unnormalize_std(value, mean, std):
	v = []
	N = len(value)

	for n in range(N):
		v.append( (value[n] * std[n]) + mean[n] )

	return v

def unscale(value, range_min, range_max):
	v = []
	N = len(value)

	for n in range(N):
		v.append( (value[n] * (range_max[n] - range_min[n])) + range_min[n] )

	return v

def scale(value, range_min, range_max):
	r = range_max - range_min
	return (value - range_min) / r

def load_image(path, type='L', resolution=-1):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		img = img.convert(type)

		if resolution > 0:
			img = img.resize( (resolution, resolution), Image.NEAREST )

		return img

