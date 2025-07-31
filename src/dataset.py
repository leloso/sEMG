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
        normalize: bool = False,
        subset: Optional[float] = None):

        self.data_file = data_file
        self.idxs = idxs
        self.load_in_memory = load_in_memory
        self.subset = subset

        if self.load_in_memory == True:
            self.samples, self.labels = self._load_data(self.idxs)
            if normalize == True:
                self._preprocess_data()

        else:
            self.f = h5py.File(self.data_file, 'r')

            if self.subset is not None:
                size = round(subset * len(self.idxs))
                self.idxs = np.random.choice(self.idxs, size, replace=False)

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

            if self.subset is not None:

                num_to_keep = int(samples.shape[0] * self.subset)

                random_indices = np.random.choice(samples.shape[0], size=num_to_keep, replace=False)

                samples = samples[random_indices, :, :]

                labels = labels[random_indices]
 
            return samples, labels
    
    def _preprocess_data(self):
        """
        Applies per-sample, per-channel normalization to the loaded data and converts to torch tensor.
        """
        normalized_samples = []
        print("Applying per-sample, per-channel normalization...")
        for sample in self.samples:
            # sample shape is (timesteps, channels)
            # Calculate mean and std for each channel within this sample
            mean_per_channel = np.mean(sample, axis=0, keepdims=True) # shape (1, channels)
            std_per_channel = np.std(sample, axis=0, keepdims=True)   # shape (1, channels)

            # Add a small epsilon to avoid division by zero
            std_per_channel[std_per_channel == 0] = 1e-6

            normalized_sample = (sample - mean_per_channel) / std_per_channel
            normalized_samples.append(normalized_sample)

        self.samples = np.stack(normalized_samples) # Stack the processed samples back into a single numpy array
        self.samples = torch.from_numpy(self.samples).float() # Convert to torch tensor
        self.labels = torch.from_numpy(self.labels).long() # Convert labels to torch tensor

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return self.samples.shape[0] if self.load_in_memory == True else len(self.idxs) 

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
            i = self.idxs[idx]
            sample = self.f['samples'][i]
            label = self.f['labels'][i]
            

        # Assuming labels are already in a suitable format (e.g., integers)
        return sample, label
    
    def __del__(self):
        """Destructor to clean up memory when dataset is deleted"""
        if hasattr(self, 'samples'):            
            # Delete the array
            del self.samples
            
            # Force garbage collection
            gc.collect()


class DomainAdaptationDataset(BaseDataset):
    """Combined dataset for DA training"""
    def __init__(self, data_file, idxs, target_idxs, load_in_memory=True, normalize: bool = False):
        
        """
        Args:
            source_data: [N_source, channels, time_samples] - Source sEMG
            source_labels: [N_source] - Source gesture labels
            target_data: [N_target, channels, time_samples] - Target sEMG (unlabeled)
        """
        super().__init__(idxs=idxs, data_file=data_file, load_in_memory=load_in_memory)
        
        self.target_idxs = target_idxs
        if self.load_in_memory == True:
            self.target_samples, self.target_labels = self._load_data(self.target_idxs)
            if normalize == True:
                self.target_samples = self._preprocess_data(self.target_samples)

            self.length = max(len(self.target_idxs), len(self.idxs))
        
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

    data_file = 'processed_data/data.h5'

    idxs = filter_data_from_h5(
        data_file=data_file,
        subjects=[1, 2, 3, 4, 5, 6],  
        positions=[0, 1, 2, 3, 4],
        sessions=0
    )

    dataset = BaseDataset(data_file, idxs=idxs)
    print(f"Dataset size: {len(dataset)}")

    print(dataset[0][0].shape)