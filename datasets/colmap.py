import os
import math
import torch
import numpy as np

from PIL import Image
from ._utils import *

from torch.utils.data import Dataset, DataLoader
from pyquaternion import Quaternion


class ColmapDataset(Dataset):
	"""
	Datasets with camera poses created by the COLMAP SfM tool:

		https://demuc.de/colmap/
		https://colmap.github.io/

	It's expected that the dataset's root dir contains an image sequence,
	along with the images.txt exported from COLMAP of the groundtruth. 
	"""

	def __init__(self, root_dir, type='train', input_channels=3, input_resolution=(224,224), 
			   normalize_outputs=True, scale_outputs=True, relative_pose=False, 
			   predict_orientations=False, transform=None):
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

		self.relative_pose        = relative_pose
		self.predict_orientations = predict_orientations
		self.input_channels       = input_channels
		self.input_resolution     = input_resolution
		self.normalize_outputs    = normalize_outputs
		self.scale_outputs 	      = scale_outputs

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

		# recursively load metadata
		self.search_directory(root_dir)

		# calc stats for train
		if type == 'train':
			calc_dataset_stats(self)

	def coordinate_space(self):
		return { 
			"x": 0,  # x coordinate
			"y": 1,  # y coordinate (height)
			"z": 2   # z coordinate (depth)
		}

	def output_dims(self):
		if self.predict_orientations:
			return 7	 # translation + orientation
		else:
			return 3	 # translation (or position)
			
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

		if self.relative_pose:
			return self.load_relative(img_set, img_num)
		else:
			return self.load_absolute(img_set, img_num)

	def load_absolute(self, img_set, img_num):
		# load and format the images, depending on number of channels
		img_type = 'RGB' if self.input_channels == 3 else 'L'
		img = load_image(self.images[img_set][img_num], type=img_type, resolution=self.input_resolution)

		# apply image transform
		if self.transform is not None:
			img = self.transform(img)
		else:
			img = torch.from_numpy(img.transpose((2, 0, 1)))

		# scale/normalize output
		pose = self.poses[img_set][img_num][:self.output_dims()]

		if self.scale_outputs:
			pose = scale(pose, self.output_range)

		if self.normalize_outputs:
			pose = normalize_std(pose, self.output_mean, self.output_std)

		#print('idx {:04d}  {:s}'.format(idx, str(pose)))
		return img, torch.Tensor(pose)

	def load_relative(self, img_set, img_num):
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

		quat_delta = quat_1.inverse * quat_2				 # relative rotation

		if quat_delta == False:
			print('warning:  zero quaternion, relative rotation ({:d})'.format(idx))

		translation = [pose_2[n] - pose_1[n] for n in range(3)] # global translation
		translation = quat_1.rotate(translation)			 # relative translation
	
		# combine into [translation, orientation]
		pose_delta = translation

		if self.predict_orientations:
			pose_delta += [quat_delta.x, quat_delta.y, quat_delta.z, quat_delta.w]

		# scale/normalize output
		if self.scale_outputs:
			pose_delta = scale(pose_delta, self.output_range)

		if self.normalize_outputs:
			pose_delta = normalize_std(pose_delta, self.output_mean, self.output_std)

		#print('idx {:04d}  {:s}'.format(idx, str(pose_delta)))
		return img, torch.Tensor(pose_delta)

	def unnormalize(self, value, type='output'):
		if type != 'output':
			Exception('type must be output')

		if self.normalize_outputs:
			value = unnormalize_std(value, self.output_mean, self.output_std)

		if self.scale_outputs:
			value = unscale(value, self.output_range)

		return value 

	def initial_pose(self):
		pose = self.poses[0][0]
		return [pose[0], pose[1], pose[2]], [pose[3], pose[4], pose[5], pose[6]]	# position, orientation

	def pose_update(self, pose, delta):
		(position, orientation) = pose

		# relative pose (add delta)
		if self.relative_pose:
			if self.predict_orientations:
				prev_quat = Quaternion(orientation[3], orientation[0], orientation[1], orientation[2])

				translation = [delta[0], delta[1], delta[2]]					# relative translation
				delta_quat  = Quaternion(delta[6], delta[3], delta[4], delta[5])	# relative rotation		

				translation_rotated = prev_quat.inverse.rotate(translation)		# relative -> global translation
				next_quat = prev_quat * delta_quat 						# relative -> global rotation

				position = vector_add(position, translation_rotated)
				orientation = [next_quat.x, next_quat.y, next_quat.z, next_quat.w] 
			else:
				translation = [delta[0], delta[1], delta[2]]
				position = vector_add(position, translation)
				orientation = [0.0, 0.0, 0.0, 0.0]
		else:
			# absolute pose estimation
			position = [delta[0], delta[1], delta[2]]

			if self.predict_orientations:
				orientation = [delta[3], delta[4], delta[5], delta[6]]
			else:
				orientation = [0.0, 0.0, 0.0, 0.0]

		return position, orientation 
 
	def load_stats(self, dataset):
		self.input_mean   = dataset.input_mean
		self.input_std    = dataset.input_std
		self.output_range = dataset.output_range
		self.output_mean  = dataset.output_mean
		self.output_std   = dataset.output_std
		
	def save_stats(self, filename):
		save_dataset_stats(self, filename)

	def search_directory(self, path):
		#print('searching ' + path)

		if not self.load_directory(path):
			for subdir in os.listdir(path):
				subdir_path = os.path.join(path, subdir)
				if os.path.isdir(subdir_path):
					self.search_directory(subdir_path)

	def load_directory(self, path):
		pose_list_filename = os.path.join(path, "images.txt")

		if not os.path.isfile(pose_list_filename):
			return False

		pose_file = open(pose_list_filename, 'r')
		
		dir_poses = []
		dir_images = []

		while(True):
			pose_line = pose_file.readline()

			if not pose_line:
				break

			if pose_line[0] == '#':
				continue

			#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
			pose_tokens = pose_line.rstrip().split(' ')
			
			if len(pose_tokens) != 10:
				print('invalid pose: ' + pose_line)
				break

			# parse tokens
			image_id = int(pose_tokens[0])
			camera_id = int(pose_tokens[8])
			image_name = pose_tokens[9]

			# https://colmap.github.io/format.html#images-txt
			R = (float(pose_tokens[1]), float(pose_tokens[2]), float(pose_tokens[3]), float(pose_tokens[4]))
			t = (float(pose_tokens[5]), float(pose_tokens[6]), float(pose_tokens[7]))

			# get the camera center from -R^t * T
			q = Quaternion(w=R[0], x=R[1], y=R[2], z=R[3]).inverse
			t = np.negative(q.rotate(np.array([t[0], t[1], t[2]]))).tolist()

			# (this is the same as transposing the quaternion's rotation matrix)
			#m = Quaternion(w=R[0], x=R[1], y=R[2], z=R[3]).rotation_matrix.transpose()
			#t = np.negative(np.dot(m, np.array([t[0], t[1], t[2]]))).tolist()

			# store metadata
			dir_poses.append(t + [q.x, q.y, q.z, q.w])
			dir_images.append(os.path.join(self.root_dir, image_name))

			# skip points2D line
			pose_file.readline()

		pose_file.close()

		self.poses.append(dir_poses)
		self.images.append(dir_images)

		self.num_images += len(dir_images)
		print('({:s}) found {:04d} images under {:s}'.format(self.type, len(dir_images), path))

		return True

