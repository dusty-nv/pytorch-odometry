import os
import math
import torch
import numpy

from torch.utils.data import Dataset, DataLoader
from PIL import Image

class TUMSlamDataset(Dataset):
	"""https://vision.in.tum.de/data/datasets/rgbd-dataset"""
	"""https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html"""

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

		# TODO split train/val
		print('({:s}) searching {:s}'.format(type, root_dir))
		self.search_directory(root_dir)
		print('({:s}) found {:d} images under {:s}'.format(type, self.num_images, root_dir))
		
	def output_dims(self):
		# https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats#ground-truth_trajectories
		return 7	# x, y, z, qx, qy, qz, qw
			
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

		img_1 = load_image(self.images[img_set][img_num])
		img_2 = load_image(self.images[img_set][img_num+1])
		img_0 = Image.new('L', img_1.size, (0))
		#img_0 = Image.fromarray(numpy.asarray(img_2) - numpy.asarray(img_1))

		img = Image.merge("RGB", (img_1, img_2, img_0))	# TODO try img_2-img_1 for third band?

		if self.transform is not None:
			img = self.transform(img)
		
		# calc pose deltas
		pose_1 = self.poses[img_set][img_num]
		pose_2 = self.poses[img_set][img_num+1]

		pose_delta = [b - a for a, b in zip(pose_1, pose_2)]

		#print('idx {:04d}  {:s}'.format(idx, str(pose_delta)))
		return img, torch.Tensor(pose_delta)

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

		print('({:s}) found {:d} images under {:s}'.format(self.type, len(dir_images), path))
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

def load_image(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')		# change to grayscale?
