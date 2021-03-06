
import torchvision.models

from .vggx import *
from .resnetx import *
from .reshape import reshape_model

def get_model_names():
	names = [name for name in torchvision.models.__dict__
		    if name.islower() and not name.startswith("__")
		    and callable(torchvision.models.__dict__[name])]

	names.append("resnet18x")
	names.append("resnet50x")
	names.append("vgg8x")
	names.append("vgg8x_1024")
	names.append("vgg11x")
     
	return sorted(names)


def create_model(arch, pretrained=True, input_channels=3, outputs=1000):

	def pretrained_available(arch, pretrained, input_channels):
		if pretrained and input_channels != 3:
			print("warning:  attempted to use pre-trained '{:s}' model with {:d} input channels".format(arch, input_channels))
			print("          pre-trained '{:s}' is only available for 3 input channels".format(arch))
			print("          proceeding to create '{:s}' without pre-trained model".format(arch))
			return False

		return pretrained

	if arch == "resnet18x":
		pretrained = pretrained_available(arch, pretrained, input_channels)
		model = resnet18x(in_channels=input_channels, pretrained=pretrained)

	elif arch == "resnet50x":
		pretrained = pretrained_available(arch, pretrained, input_channels)
		model = resnet50x(in_channels=input_channels, pretrained=pretrained)

	elif arch.startswith("vgg8x"):
		if pretrained:
			print("warning:  pre-trained vgg8x not available, proceeding without pre-trained model")
			pretrained = False

		fc_features = 4096

		if arch == "vgg8x_1024":
			fc_features = 1024

		model = vgg8x(in_channels=input_channels, fc_features=fc_features, pretrained=pretrained)

	elif arch == "vgg11x":
		pretrained = pretrained_available(arch, pretrained, input_channels)
		model = vgg11x(in_channels=input_channels, pretrained=pretrained)

	else:
		if input_channels != 3:
			raise Exception("cannot create '{:s}' model with {:d} input channels (only 3)".format(arch, input_channels))

		model = torchvision.models.__dict__[arch](pretrained=pretrained)

	if outputs != 1000:
		model = reshape_model(model, arch, outputs)

	return model




