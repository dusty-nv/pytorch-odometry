#
# converts a saved PyTorch model to ONNX format
# 
import os
import torch
import argparse

from models import *

model_names = get_model_names()

# parse command line
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='model_best.pth.tar', help="path to input PyTorch model (default: model_best.pth.tar)")
parser.add_argument('--output', type=str, default='', help="desired path of converted ONNX model (default: <ARCH>.onnx)")
parser.add_argument('--model-dir', type=str, default='', help="directory to look for the input PyTorch model in, and export the converted ONNX model to (if --output doesn't specify a directory)")
parser.add_argument('--add-softmax', action='store_true', help="disable adding nn.Softmax layer to model (default is to not add Softmax)")

opt = parser.parse_args() 
print(opt)

# format input model path
if opt.model_dir:
	opt.model_dir = os.path.expanduser(opt.model_dir)
	opt.input = os.path.join(opt.model_dir, opt.input)

# set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('running on device ' + str(device))

# load the model checkpoint
print('loading checkpoint:  ' + opt.input)
checkpoint = torch.load(opt.input)

arch = checkpoint['arch']
input_channels = checkpoint['input_channels']
output_dims = checkpoint['output_dims']

if 'resolution' in checkpoint:
	resolution = (checkpoint['resolution'], checkpoint['resolution'])
else:
	resolution = (checkpoint['height'], checkpoint['width'])

if 'pretrained' in checkpoint:
	pretrained = checkpoint['pretrained']
else:
	pretrained = False

# create the model architecture
print('using model:    ' + arch)
print('model epoch:    ' + str(checkpoint['epoch']))
print('model loss:     ' + str(checkpoint['best_loss']))
print('model path_err: ' + str(checkpoint['path_error']) if 'path_error' in checkpoint else '0.0')
print('model inputs:   {:d}x{:d}x{:d}'.format(resolution[1], resolution[0], input_channels))
print('model outputs:  {:d}'.format(output_dims))

model = create_model(arch, pretrained=pretrained, input_channels=input_channels, outputs=output_dims)

# load the model weights
model.load_state_dict(checkpoint['state_dict'])

# add softmax layer
if opt.add_softmax:
	print('adding nn.Softmax layer to model...')
	model = torch.nn.Sequential(model, torch.nn.Softmax(1))

model.to(device)
model.eval()

print(model)

# create example image data
input = torch.ones((1, input_channels, resolution[0], resolution[1])).cuda()
print('input size:  {:d}x{:d}x{:d}'.format(resolution[1], resolution[0], input_channels))

# format output model path
if not opt.output:
	opt.output = os.path.join(opt.model_dir if opt.model_dir else os.path.dirname(opt.input), arch + '.onnx')
elif opt.model_dir and opt.output.find('/') == -1 and opt.output.find('\\') == -1:
	opt.output = os.path.join(opt.model_dir, opt.output)

# export the model
input_names = [ "input_0" ]
output_names = [ "output_0" ]

print('exporting model to ONNX...')
torch.onnx.export(model, input, opt.output, verbose=True, input_names=input_names, output_names=output_names)
print('model exported to:  {:s}'.format(opt.output))


