import numpy as np
from numpy.typing import NDArray


class MemmapConcatenator:
    """
    Concatenates multiple memmaps into a single array-like object without loading the individual files into memory.    
    """
    def __init__(self, memmap_list: list[np.memmap], context_length: int):
        """
    Concatenates multiple memmaps into a single array-like object without loading the individual files into memory.  
        Args:
            memmap_list: List of memmap objects to concatenate. Only 1D memmaps are supported.
        """
        self.memmap_list = memmap_list
        # Cummulative length of all files in tokens without the last context_length
        self.lengths: NDArray[np.int64] = np.array([memmap.shape[0]-context_length for memmap in memmap_list]).cumsum()
        self.offsets = [0, *self.lengths.tolist()]
        self.context_length = context_length
        self.full_length = self.lengths[-1] + context_length

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.full_length)
            file = self.get_dataset_idx(start)
            ret = self.memmap_list[file][start - self.offsets[file]:stop - self.offsets[file]:step]
            return ret
        elif isinstance(idx, int):
            file = self.get_dataset_idx(idx)
            offset = self.offsets[file]
            return self.memmap_list[file][idx - offset]
        else:
            raise IndexError("Only integer and slice indexing are supported.")
        
    def get_dataset_idx(self, idx: int) -> int:
        return self.lengths.searchsorted(idx, side='right')
    
    def __len__(self):
        # Return the total number of tokens in all memmaps without the last context_length
        return self.lengths[-1]

