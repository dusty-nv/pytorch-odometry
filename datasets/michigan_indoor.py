import os
import math
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MichiganIndoorDataset(Dataset):
	"""https://deepblue.lib.umich.edu/data/concern/data_sets/3t945q88k"""

	def __init__(self, root_dir, type='train', transform=None):
		"""
		Args:
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.type = type
		self.root_dir = root_dir		
		self.transform = transform
		self.num_images = 0

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
				continue

			print('{:s} - {:d} images'.format(dir_path, len(dir_images)))

			self.images.append(dir_images)
			self.poses.append(dir_poses)

			self.num_images += len(dir_images) - 1

		print(str(self.num_images))
		#print(self.images)
		
	def output_dims(self):
		return 2   # speed delta, theta delta     #3	# x, y, theta
			
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

		rgb_diff = True

		if rgb_diff:
			img_1 = load_image(self.images[img_set][img_num], 'RGB')
			img_2 = load_image(self.images[img_set][img_num+1], 'RGB')
			
			img = np.asarray(img_2).astype(np.float32) - np.asarray(img_1).astype(np.float32) #Image.fromarray(np.asarray(img_2) - np.asarray(img_1))
			img = (img + 255.0) / 2.0
			img = Image.fromarray(img.astype(np.uint8))		
		else:
			img_1 = load_image(self.images[img_set][img_num])
			img_2 = load_image(self.images[img_set][img_num+1])
			img_0 = Image.new('L', img_1.size, (0))
			#img_0 = Image.fromarray(np.asarray(img_2) - np.asarray(img_1))
			img = Image.merge("RGB", (img_1, img_2, img_0))

		if self.transform is not None:
			img = self.transform(img)
		
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

		return img, torch.Tensor(pose_delta)


def load_image(path, type='L'):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(type)
