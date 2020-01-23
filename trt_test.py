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

from datasets import create_dataset, get_dataset_names

# parse command line
parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', default='tum', help='dataset type: tum, icl, michigan (default: tum)')
parser.add_argument('--model', type=str, default='model_trt.pth', help="path to input PyTorch model (default: model_best.pth.tar)")
parser.add_argument('--plot', type=str, default='path.jpg', help="filename of the path plot")
parser.add_argument('-b', '--batch-size', default=1, type=int)
parser.add_argument('-j', '--workers', default=1, type=int)
parser.add_argument('--resolution', default=224, type=int, metavar='N')
parser.add_argument('--width', default=0, type=int)
parser.add_argument('--height', default=0, type=int)
parser.add_argument('--input-channels', default=2, type=int, dest='input_channels')
parser.add_argument('--scale-outputs', dest='scale_outputs', action='store_true', help='scale network outputs (default: True)')
parser.add_argument('--no-scale-outputs', dest='scale_outputs', action='store_false', help='do not scale network outputs')
parser.add_argument('--normalize-outputs', dest='normalize_outputs', action='store_true', help='normalize network outputs (default: True)')
parser.add_argument('--no-normalize-outputs', dest='normalize_outputs', action='store_false', help='do not normalize network outputs')
parser.add_argument('--relative-pose', dest='relative_pose', action='store_true', help='compute Relative Pose Estimation (default: True)')
parser.add_argument('--absolute-pose', dest='relative_pose', action='store_false', help='compute Absolute Pose Estimation')
parser.add_argument('--predict-orientations', dest='predict_orientations', action='store_true', help='predict pose orientations (default: True)')

parser.set_defaults(scale_outputs=True)
parser.set_defaults(normalize_outputs=True)
parser.set_defaults(relative_pose=True)
parser.set_defaults(predict_orientations=True)

args = parser.parse_args() 

if args.width <= 0:
	args.width = args.resolution

if args.height <= 0:
	args.height = args.resolution

print(args)


# load the dataset
dataset = create_dataset(args.dataset, root_dir=args.data, type='train', input_channels=args.input_channels, input_resolution=(args.width, args.height), scale_outputs=args.scale_outputs, normalize_outputs=args.normalize_outputs, relative_pose=args.relative_pose, predict_orientations=args.predict_orientations)

normalize = transforms.Normalize(mean=dataset.input_mean,
                                std=dataset.input_std)

dataset.transform = transforms.Compose([
       #transforms.Resize((args.height, args.width), interpolation=PIL.Image.NEAREST),
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


# distance function
def distance(a, b):
	d = 0.0
	for n in range(len(a)):
		d += math.pow(b[n] - a[n], 2)
	return math.sqrt(d)


# compute the trajectory
position, orientation = dataset.initial_pose()
position_gt, orientation_gt = dataset.initial_pose()

position_history = [position]
position_history_gt = [position_gt]


for i, (images, target) in enumerate(dataloader):
	images = images.cuda()
	output = model(images)

	batch_size = len(output)

	for n in range(batch_size):
		output_unnorm = dataset.unnormalize(output.cpu().numpy()[n])
		target_unnorm = dataset.unnormalize(target.cpu().numpy()[n])

		#print("{:d} out={:s} gt={:s}".format(i, str(output_unnorm), str(target_unnorm)))

		try:
			# update the vel/heading pose
			position, orientation = dataset.pose_update((position, orientation), output_unnorm)
			position_gt, orientation_gt = dataset.pose_update((position_gt, orientation_gt), target_unnorm)
		except:
			# early on in training, exceptions may fly from invalid quaternion operations
			# from erroneous network outputs - ignore them, they should disappear over time
			print('exception: ' + str(sys.exc_info()[1]))

		print("{:04d} pos={:s} gt={:s} err={:f}".format(i, str(position), str(position_gt), distance(position, position_gt)))

		# add to position history
		position_history.append(position)
		position_history_gt.append(position_gt)



# compute the path error
path_error = distance(position_history[-1], position_history_gt[-1])
print('path error = {:f}'.format(path_error))

# plot the path data
position_history = list(map(list,zip(*position_history)))	
position_history_gt = list(map(list,zip(*position_history_gt)))	

# create the plots (X/Z and Y, if available)
num_plots = 2 if len(position_history) > 2 else 1
fig, plots = plt.subplots(1, num_plots, figsize=(20, 10))

# retrieve the coordinate space mapping
cs = dataset.coordinate_space()

# plot the trajectory data
plots[0].plot(position_history[cs["x"]], position_history[cs["z"]], 'b--', label=args.model)
plots[0].plot(position_history_gt[cs["x"]], position_history_gt[cs["z"]], 'r--', label='groundtruth')
plots[0].set_xlabel("X (meters)")
plots[0].set_ylabel("Z (meters)")

if num_plots > 1:
	plots[1].plot(position_history[cs["y"]], 'b--', label=args.model)
	plots[1].plot(position_history_gt[cs["y"]], 'r--', label='groundtruth')
	plots[1].set_xlabel("Frame Number")
	plots[1].set_ylabel("Y (meters)")

# set the figure title
plt.suptitle('(path_error={:f}, N={:d})'.format(path_error, len(dataset)))
fig.legend(loc="upper right")

# save the figure
fig.savefig(args.plot)
print('saved path plot to:  ' + args.plot)




