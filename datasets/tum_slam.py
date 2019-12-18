import os
import math
import torch
import numpy as np

from PIL import Image
from ._utils import *

from torch.utils.data import Dataset, DataLoader
from pyquaternion import Quaternion


class TUMSlamDataset(Dataset):
	"""https://vision.in.tum.de/data/datasets/rgbd-dataset"""
	"""https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html"""

	def __init__(self, root_dir, type='train', input_channels=3, input_resolution=(224,224), 
                  normalize_output=True, scale_output=True, transform=None):
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

		self.input_channels   = input_channels
		self.input_resolution = input_resolution
		self.normalize_output = normalize_output
		self.scale_output 	  = scale_output

		self.input_mean = [0.37613192, 0.37610024]
		self.input_std = [0.19967583, 0.19966353]

		if self.input_channels == 3:
			self.input_mean.append(0.0)
			self.input_std.append(1.0)
		elif self.input_channels == 6:
			self.input_mean = [0.387192, 0.37493756, 0.37024707, 0.38716564, 0.37490633, 0.3701989]
			self.input_std = [0.18855667, 0.20194934, 0.23527457, 0.18855427, 0.20193432, 0.23524201]

		self.output_range = [[-0.020121246576309204, 0.20444628596305847], [-0.4495636522769928, 0.005035788752138615], [-0.27828314900398254, 0.03655131906270981], [-1.0000364780426025, 1.0001380443572998], [-0.058813586831092834, 0.06137028709053993], [-0.139356330037117, 0.23180893063545227], [-0.019846243783831596, 0.019199304282665253]] #[[-1.0, 1.0] for n in range(self.output_dims())]
		self.output_mean = [0.08903043, 0.97326106, 0.88268435, 0.99107015, 0.48926163, 0.3753603, 0.50792235]
		self.output_std = [0.02054254, 0.01464775, 0.02503456, 0.06498827, 0.03982857, 0.01947158, 0.03847046]

		# TODO split train/val
		print('({:s}) searching {:s}'.format(type, root_dir))
		self.search_directory(root_dir)
		print('({:s}) found {:d} images under {:s}'.format(type, self.num_images, root_dir))
		
	def output_dims(self):
		# https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats#ground-truth_trajectories
		return 7	# tx, ty, tz, qx, qy, qz, qw
			
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

		quat_1 = Quaternion(pose_1[6], pose_1[3], pose_1[4], pose_1[5])
		quat_2 = Quaternion(pose_2[6], pose_2[3], pose_2[4], pose_2[5])

		quat_delta = quat_1.inverse * quat_2

		xyz_delta = [pose_2[n] - pose_1[n] for n in range(3)]
		xyz_delta = quat_1.rotate(xyz_delta)
		#xyz_speed = math.sqrt(xyz_delta[0] * xyz_delta[0] + xyz_delta[1] * xyz_delta[1] + xyz_delta[2] * xyz_delta[2])

		pose_delta = xyz_delta + [quat_delta.x, quat_delta.y, quat_delta.z, quat_delta.w] #quat_delta.q.tolist()

		# scale/normalize output
		if self.scale_output:
			pose_delta = scale(pose_delta, self.output_range)

		if self.normalize_output:
			pose_delta = normalize_std(pose_delta, self.output_mean, self.output_std)

		#print('idx {:04d}  {:s}'.format(idx, str(pose_delta)))
		return img, torch.Tensor(pose_delta)

	def unnormalize(self, value, type='output'):
		if type != 'output':
			Exception('type must be output')

		return unscale(unnormalize_std(value, self.output_mean, self.output_std), self.output_range)

	def initial_pose(self):
		pose = self.poses[0][0]
		return [pose[3], pose[4], pose[5], pose[6]]

	def pose_update(self, pose, delta):
		prev_quat = Quaternion(pose[3], pose[0], pose[1], pose[2])

		delta_xyz  = [delta[0], delta[1], delta[2]]
		delta_quat = Quaternion(delta[6], delta[3], delta[4], delta[5])

		delta_xyz_rotated = prev_quat.rotate(delta_xyz)
		next_quat = prev_quat * delta_quat

		return [next_quat.x, next_quat.y, next_quat.z, next_quat.w], delta_xyz_rotated

	def search_directory(self, path):
		#print('searching ' + path)

		if not self.load_directory(path):
			for subdir in os.listdir(path):
				subdir_path = os.path.join(path, subdir)
				if os.path.isdir(subdir_path):
					self.search_directory(subdir_path)

	def load_directory(self, path):
		rgb_list_filename = os.path.join(path, "rgb.txt")
		pose_list_filename = os.path.join(path, "groundtruth.txt")

		if not os.path.isfile(rgb_list_filename) or not os.path.isfile(pose_list_filename):
			return False

		rgb_list = self.read_file_list(rgb_list_filename)
		pose_list = self.read_file_list(pose_list_filename)
		
		matches = self.associate(rgb_list, pose_list, offset=0.0, max_difference=0.02)

		dir_images = []
		dir_poses = []

		for rgb_key, pose_key in matches:
			#print('{:f} {:s} => {:f} {:s}'.format(rgb_key, rgb_list[rgb_key][0], pose_key, str(pose_list[pose_key])))
			dir_images.append(os.path.join(path, rgb_list[rgb_key][0]))
			dir_poses.append([float(i) for i in pose_list[pose_key]])

		if len(dir_images) != len(dir_poses):
			print('dataset {:s} length mismatch - {:d} images vs. {:d} poses'.format(path, len(dir_images), len(dir_poses)))
			return True

		print('({:s}) found {:04d} images under {:s}'.format(self.type, len(dir_images), path))
		self.num_images += len(dir_images)

		self.images.append(dir_images)
		self.poses.append(dir_poses)

		return True

	def read_file_list(self, filename):
		"""
		Reads a trajectory from a text file. 

		File format:
		The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
		and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 

		Input:
		filename -- File name

		Output:
		dict -- dictionary of (stamp,data) tuples

		"""
		file = open(filename)
		data = file.read()
		lines = data.replace(","," ").replace("\t"," ").split("\n") 
		list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
		list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
		return dict(list)

	def associate(self, first_list, second_list, offset, max_difference):
		"""
		Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
		to find the closest match for every input tuple.

		Input:
		first_list -- first dictionary of (stamp,data) tuples
		second_list -- second dictionary of (stamp,data) tuples
		offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
		max_difference -- search radius for candidate generation

		Output:
		matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))

		"""
		first_keys = list(first_list) #first_list.keys()
		second_keys = list(second_list) #second_list.keys()
		potential_matches = [(abs(a - (b + offset)), a, b) 
						for a in first_keys 
						for b in second_keys 
						if abs(a - (b + offset)) < max_difference]
		potential_matches.sort()
		matches = []
		for diff, a, b in potential_matches:
			if a in first_keys and b in second_keys:
				first_keys.remove(a)
				second_keys.remove(b)
				matches.append((a, b))

		matches.sort()
		return matches

