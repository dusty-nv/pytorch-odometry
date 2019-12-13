
import torchvision.models

from .vggx import vgg11x, vgg8x
from .reshape import reshape_model

def get_model_names():
	names = [name for name in torchvision.models.__dict__
		    if name.islower() and not name.startswith("__")
		    and callable(torchvision.models.__dict__[name])]

	names += "vgg8x"
	names += "vgg11x"
     
	return sorted(names)


def create_model(arch, pretrained=True, input_channels=3, outputs=1000):
	if arch == "vgg8x":
		if pretrained:
			print("warning:  pre-trained vgg8x not available, proceeding without pre-trained model")
			pretrained = False

		model = vgg8x(in_channels=input_channels, pretrained=pretrained)

	elif arch == "vgg11x":
		if pretrained and input_channels != 3:
			print("warning:  attempted to use pre-trained '{:s}' model with {:d} input channels".format(arch, input_channels))
			print("          pre-trained '{:s}' is only available for 3 input channels".format(arch))
			print("          proceeding to create '{:s}' without pre-trained model")

			pretrained = False

		model = vgg11x(in_channels=input_channels, pretrained=pretrained)

	else:
		if input_channels != 3:
			raise Exception("cannot create '{:s}' model with {:d} input channels (only 3)".format(arch, input_channels))

		model = torchvision.models.__dict__[args.arch](pretrained=pretrained)

	if outputs != 1000:
		model = reshape_model(model, arch, outputs)

	return model




