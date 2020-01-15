
import torch.utils.data

from copy import deepcopy

from .colmap import ColmapDataset
from .michigan_indoor import MichiganIndoorDataset
from .tum_slam import TUMSlamDataset


def get_dataset_names():
	return ["tum", "icl", "michigan", "colmap"]

def create_dataset(name, **kwargs):
	if name == "tum" or name == "icl":
		return TUMSlamDataset(**kwargs)
	elif name == "michigan":
		return MichiganIndoorDataset(**kwargs)
	elif name == "colmap":
		return ColmapDataset(**kwargs)


class DataLoaderCache(torch.utils.data.DataLoader):

	def __init__(self, dataset, in_memory=False, **kwargs):
		#print("DataLoaderCache.__init__() " + str(kwargs))
		super(DataLoaderCache, self).__init__(dataset, **kwargs)
		self.in_memory = in_memory
		self.data_cache = []

		if in_memory:
			self.load_all()

	def load_all(self):
		#super_iter = super(DataLoaderCache, self).__iter__()
		
		temp_loader = torch.utils.data.DataLoader(self.dataset, 
					batch_size=1, num_workers=self.num_workers)
						
		print("DataLoaderCache -- loading dataset into memory")

		# https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
		for i, (sample, target) in enumerate(temp_loader):
			sample_copy = deepcopy(sample[0])
			target_copy = deepcopy(target[0])
			self.data_cache.append((sample_copy, target_copy))
			del sample

		#for n in range(len(self.dataset)):
		#	self.data_cache.append(self.dataset.__getitem__(n))

		print("DataLoaderCache -- loaded {:d} data entries".format(len(self.data_cache)))

	def __iter__(self):
		#print("DataLoaderCache -- __iter__()")
		if self.in_memory:
			return _DataLoaderCacheIter(self)
		else:
			return super(DataLoaderCache, self).__iter__()


# https://github.com/pytorch/pytorch/blob/54a63e0420c2810930d848842142a07071204c06/torch/utils/data/dataloader.py#L300
class _DataLoaderCacheIter(object):
	def __init__(self, loader):
		#print("_DataLoaderCacheIter.__init__()")
		self.loader = loader
		self.index_sampler = loader._index_sampler
		self.sampler_iter = iter(self.index_sampler)

	def __iter__(self):
		return self

	def _next_index(self):
		return next(self.sampler_iter)  # may raise StopIteration

	def __next__(self):
		index = self._next_index()  # may raise StopIteration
		#print('DataLoaderCacheItr.__next__() ' + str(index))

		data = []

		for n in index:
			data.append(self.loader.data_cache[n])

		data = self.loader.collate_fn(data)

		#if self._pin_memory:
		#    data = _utils.pin_memory.pin_memory(data)

		return data

	def __len__(self):
		n = len(self.index_sampler)
		#print('DataLoaderCacheIter.__len__  = ' + str(n))
		return n

	def __getstate__(self):
		# TODO: add limited pickling support for sharing an iterator
		# across multiple threads for HOGWILD.
		# Probably the best way to do this is by moving the sample pushing
		# to a separate thread and then just sharing the data queue
		# but signalling the end is tricky without a non-blocking API
		raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)


