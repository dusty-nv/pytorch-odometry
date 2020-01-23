#
# converts a saved PyTorch model to TensorRT format
# 
import os
import torch
import argparse
import tensorrt as trt

from torch2trt import torch2trt
from models import *


# parse command line
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='model_best.pth.tar', help="path to input PyTorch model (default: model_best.pth.tar)")
parser.add_argument('--output', type=str, default='', help="desired path of converted TensorRT engine (default: <ARCH>.engine)")
parser.add_argument('--model-dir', type=str, default='', help="directory to look for the input PyTorch model in, and export the converted ONNX model to (if --output doesn't specify a directory)")
parser.add_argument('--mode', type=str, default='fp16', choices=['fp16', 'fp32', 'int8'])
parser.add_argument('--max-workspace-size', type=int, default=1024*1024*32, help="TensorRT engine builder max workspace size")

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

model.to(device)
model.eval()

#print(model)

# create example image data
x = torch.ones((1, input_channels, resolution[0], resolution[1])).cuda()
print('input size:  {:d}x{:d}x{:d}'.format(resolution[1], resolution[0], input_channels))

# set input/output names
input_names = [ "input_0" ]
output_names = [ "output_0" ]

# convert to TensorRT feeding sample data as input
fp16_mode = (opt.mode == "fp16")
int8_mode = (opt.mode == "int8")

print('creating TensorRT engine...')
print('  => fp16:  {:s}'.format(str(fp16_mode)))
print('  => int8:  {:s}'.format(str(int8_mode)))

model_trt = torch2trt(model, [x], 
				  input_names=input_names, 
				  output_names=output_names,
				  fp16_mode=fp16_mode, 
				  int8_mode=int8_mode, 
				  max_workspace_size=opt.max_workspace_size,
				  log_level=trt.Logger.VERBOSE)

print('TensorRT engine created\n')

# compare outputs
y = model(x)
y_trt = model_trt(x)

print('model output difference:  ' + str(torch.max(torch.abs(y - y_trt))))

print('PyTorch output:  ' + str(y))
print('TensorRT output: ' + str(y_trt))

# format output model path
if not opt.output:
	opt.output = os.path.join(opt.model_dir if opt.model_dir else os.path.dirname(opt.input), arch + '.engine')
elif opt.model_dir and opt.output.find('/') == -1 and opt.output.find('\\') == -1:
	opt.output = os.path.join(opt.model_dir, opt.output)

# save either the state_dict or the serialized TRT engine
ext = os.path.splitext(opt.output)[1]

if ext == '.pth' or ext == '.tar':
	print('saving torch2trt model to:  {:s}'.format(opt.output))
	torch.save(model_trt.state_dict(), opt.output)
	print('torch2trt model saved to:   {:s}'.format(opt.output))
else:
	print('saving TensorRT engine to:  {:s}'.format(opt.output))
	with open(opt.output, "wb") as output_file:
		output_file.write(model_trt.engine.serialize())
	print('TensorRT engine saved to:   {:s}'.format(opt.output))


