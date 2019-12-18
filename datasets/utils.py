

from .michigan_indoor import MichiganIndoorDataset
from .tum_slam import TUMSlamDataset


def get_dataset_names():
	return ["tum", "icl", "michigan"]

def create_dataset(name, **kwargs):
	if name == "tum" or name == "icl":
		return TUMSlamDataset(**kwargs)
	elif name == "michigan":
		return MichiganIndoorDataset(**kwargs)


