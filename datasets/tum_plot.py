import os
import math
import argparse
import collections
import numpy as np

import matplotlib.pyplot as plt

from pyquaternion import Quaternion


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

def vector_mul(a, b):
	v = []
	if isinstance(b, collections.Sequence):
		for n in range(len(a)):
			v.append(a[n] * b[n])
	else:
		for n in range(len(a)):
			v.append(a[n] * b)
	return v

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

def vector_cross(a, b):
	c = [a[1]*b[2] - a[2]*b[1],
          a[2]*b[0] - a[0]*b[2],
          a[0]*b[1] - a[1]*b[0]]

	return c

def to_degrees(a):
	v = []
	for n in range(len(a)):
		v.append(a[n] * 57.295779513082320876798154814105)
	return v

def vector_dot(a, b):
	return sum([i*j for (i, j) in zip(a, b)])


def calc_pos(pose_A, pose_B):
	pos_A = [pose_A[0], pose_A[1], pose_A[2]]
	pos_B = [pose_B[0], pose_B[1], pose_B[2]]

	quat_A = Quaternion(pose_A[6], pose_A[3], pose_A[4], pose_A[5])
	quat_B = Quaternion(pose_B[6], pose_B[3], pose_B[4], pose_B[5])

	quat_delta = quat_A.inverse * quat_B
	quat_B2 = quat_A * quat_delta

	print("\n\n")
	print("quat_A = " + str(quat_A))
	#print(" ypr_A = " + str(to_degrees(quat_A.yaw_pitch_roll)))
	print("quat_B = " + str(quat_B))
	#print(" ypr_B = " + str(to_degrees(quat_B.yaw_pitch_roll)))
	print("quat_Δ = " + str(quat_delta))
	#print(" ypr_Δ = " + str(to_degrees(quat_delta.yaw_pitch_roll)))
	print("quat_B'= " + str(quat_B2)) 

	#velocity = distance(pos_B, pos_A)
	pos_delta = vector_sub(pos_B, pos_A)
	pos_delta_rotated = quat_A.rotate(pos_delta)
	pos_delta_rotated_inv = quat_A.inverse.rotate(pos_delta_rotated)

	#print("\nvelocity = {:f}".format(velocity))
	print(" ")
	print("pos_A = " + str(pos_A))
	print("pos_B = " + str(pos_B))
	print(" ")
	print("pos_delta = " + str(pos_delta))
	print("pos_delta_rotated = " + str(pos_delta_rotated))
	print("pos_delta_rotated_inv = " + str(pos_delta_rotated_inv))

	#forward_vec  = [0.0, 0.0, 1.0]
	#rotated_vec  = quat_delta.normalised.rotate(forward_vec)
	#velocity_vec = vector_mul(rotated_vec, velocity)

	#print(" ")
	#print("forward_vec  = " + str(forward_vec))
	#print("rotated_vec  = " + str(rotated_vec))
	#print("velocity_vec = " + str(velocity_vec))

	#pos_B2  = vector_add(pos_A, velocity_vec)
	#pos_err = distance(pos_B, pos_B2)

	#print(" ")
	#print("pos_A = " + str(pos_A))
	#print("pos_B = " + str(pos_B))
	#print("pos_B'= " + str(pos_B2))
	#print(" ")
	#print("pos_error = {:f}".format(pos_err))
	#print("\n\n\n")


def read_file_list(filename):
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

def associate(first_list, second_list, offset, max_difference):
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



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='TUM-RGB trajectory calculator test')

	parser.add_argument('data', metavar='DIR', help='path to dataset')
	parser.add_argument('--rgb', default="rgb.txt", type=str)
	parser.add_argument('--gt', default="groundtruth.txt", type=str)
	parser.add_argument('--plot', default="plot.jpg", type=str)

	args = parser.parse_args()

	# load grountruth
	rgb_list  = read_file_list(os.path.join(args.data, args.rgb))
	pose_list = read_file_list(os.path.join(args.data, args.gt))

	matches = associate(rgb_list, pose_list, offset=0.0, max_difference=0.02)
	poses   = []

	for rgb_key, pose_key in matches:
		poses.append([float(i) for i in pose_list[pose_key]])

	# process
	gt_coords = []
	skip = 1

	for n in range(skip, len(poses)):
		calc_pos( poses[n-skip], poses[n] )
		gt_coords.append([poses[n][0], poses[n][1], poses[n][2]])

	if args.plot:
		gt_coords = list(map(list,zip(*gt_coords)))	# split from [[x,y,z]] -> [[x],[y],[z]]	
		fig, plots = plt.subplots(1, 2, figsize=(10, 5))		
		plots[0].plot(gt_coords[0], gt_coords[1], 'r--', label='groundtruth (XY)')
		plots[1].plot(gt_coords[2], 'r--', label='groundtruth (Z)')
		fig.suptitle(args.data)
		fig.savefig(args.plot)
	
