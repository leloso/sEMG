from typing import Optional
import gc

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset




class BaseDataset(Dataset):
    """
    Base Dataset class for loading and processing sEMG data from an HDF5 file.

    Args:
        source_indices ()
        load_in_memory (bool, optional): Whether to load the entire dataset into memory. Defaults to True.
        subset (str, optional): Optional subset of data to load. Defaults to None.
        transform (callable, optional): Optional transform to be applied to a sample.
    """
    def __init__(self, 
        data_file: str = None,
        idxs = None,
        load_in_memory: bool = True,
        subset: Optional[float] = None, 
    ):

        self.data_file = data_file
        self.idxs = idxs
        self.load_in_memory = load_in_memory
        self.subset = subset

        self.size = len(self.idxs)

        if self.load_in_memory == True:
            self.samples, self.labels = self._load_data(self.idxs)

            if self.subset is not None:
                size = round(subset * len(self.idxs))
                keep_indices = np.random.choice(len(self.idxs), size, replace=False)
                self.size = keep_indices.size
                mask = np.zeros(len(self.idxs), dtype=bool)
                mask[keep_indices] = True
                self.samples = self.samples[mask]
                self.labels = self.labels[mask]

    def _load_data(self, idxs):
        """
        Loads data from the HDF5 file based on the specified sessions, repetitions, and subjects in RAM.

        Returns:
            None
        """

        with h5py.File(self.data_file, 'r') as f:
            if len(idxs) == 0:
                raise ValueError("No indexes passed.")
            print(f"Attempting to load {idxs.shape} samples to memory...")
            samples = f['samples'][idxs]

            labels = f['labels'][idxs]
            print("Loading complete")

            return samples, labels

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return self.size

    def __getitem__(self, idx: int):
        """
        Retrieves a single sample and its label from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the sample and its label as PyTorch tensors.
        """
        if self.load_in_memory == True:

            sample = self.samples[idx]
            label = self.labels[idx]
        else:
            i = self.indices[idx]
            sample = self.data_file['samples'][i]
            label = self.data_file['labels'][i]

        # Assuming labels are already in a suitable format (e.g., integers)
        return torch.from_numpy(sample).float(), torch.tensor(label).long()
    
    def __del__(self):
        """Destructor to clean up memory when dataset is deleted"""
        if hasattr(self, 'samples'):            
            # Delete the array
            del self.samples
            
            # Force garbage collection
            gc.collect()
            print("Memory cleanup completed")


class DomainAdaptationDataset(BaseDataset):
    """Combined dataset for DA training"""
    def __init__(self, data_file, idxs, target_idxs, load_in_memory=True):
        
        """
        Args:
            source_data: [N_source, channels, time_samples] - Source sEMG
            source_labels: [N_source] - Source gesture labels
            target_data: [N_target, channels, time_samples] - Target sEMG (unlabeled)
        """
        super().__init__(idxs=idxs, data_file=data_file, load_in_memory=load_in_memory)
        
        self.target_idxs = target_idxs
        self.target_data, _ = self._load_data(self.target_idxs)


        self.length = max(len(target_idxs), len(idxs))
        
    def __getitem__(self, idx):

        # Handle index wrapping if source & target are different sizes
        source_idx = idx % len(self.idxs)
        target_idx = idx % len(self.target_idxs)
        
        # Return source-sample, label, target-sample
        return  torch.from_numpy(self.source_data[source_idx]).float(), \
                torch.tensor(self.source_labels[source_idx]).long(), \
                torch.from_numpy(self.target_data[target_idx]).float()
    
    def __len__(self):
        return len(self.idxs)




# Example usage (for testing the dataset class)
if __name__ == '__main__':

    from utils import filter_data_from_h5

    data_file = 'data.h5'

    idxs = filter_data_from_h5(
        data_file=data_file,
        subjects=[1, 2, 3, 4, 5, 6],  
        positions=[0, 1, 2, 3, 4],
        sessions=0
    )
    target_idxs = filter_data_from_h5(
        data_file=data_file,
        subjects=[1, 2, 3, 4, 5, 6], 
        positions=5,
        sessions=0
    )


    dataset = DomainAdaptationDataset(data_file, idxs=idxs, target_idxs=target_idxs)
    print(f"Dataset size: {len(dataset)}")

    source, label , target = dataset[40000] 
    print(source.shape + target.shape)
    
   