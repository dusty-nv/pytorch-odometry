
from PIL import Image


def load_image(path, type='L', resolution=(-1, -1)):
	# open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
	with open(path, 'rb') as f:
		img = Image.open(f)
		img = img.convert(type)

		if resolution[0] > 0 and resolution[1] > 0:
			img = img.resize( resolution, Image.NEAREST )

		return img

#
# vector operations
#
def normalize_std(value, mean, std):
	v = []
	N = len(value)

	for n in range(N):
		v.append((value[n] - mean[n]) / std[n])

	return v

def unnormalize_std(value, mean, std):
	v = []
	N = len(value)

	for n in range(N):
		v.append((value[n] * std[n]) + mean[n])

	return v

def scale(value, value_range):
	v = []
	N = len(value)

	for n in range(N):
		v.append((value[n] - value_range[n][0]) / (value_range[n][1] - value_range[n][0]))

	return v

def unscale(value, value_range):
	v = []
	N = len(value)

	for n in range(N):
		v.append((value[n] * (value_range[n][1] - value_range[n][0])) + value_range[n][0])

	return v

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

def vector_mul(a, b):
	v = []
	if isinstance(b, collections.Sequence):
		for n in range(len(a)):
			v.append(a[n] * b[n])
	else:
		for n in range(len(a)):
			v.append(a[n] * b)
	return v

def vector_cross(a, b):
	c = [a[1]*b[2] - a[2]*b[1],
          a[2]*b[0] - a[0]*b[2],
          a[0]*b[1] - a[1]*b[0]]

	return c

