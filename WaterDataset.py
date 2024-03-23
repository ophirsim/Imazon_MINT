import torch
from torch.utils.data import Dataset
import os, os.path
import numpy as np

class WaterDataset(Dataset):
    """Dataset class for Imazon's water data"""

    def __init__(self, data_path, timestep = 6, iterative = False, transform = lambda x: x.unsqueeze(0), target_transform = lambda x: x.unsqueeze(0), dtype=torch.float, device='cpu'):
        """
        The constructor for the WaterDataset dataset
        
        Args:
         - data_path: string -- the path to the dataset containing all of the dataset's files
         - timestep: int -- the number of months between the sample water map and the label water map
                            e.x: if timestep = 10, then the label will be a water map 10 months after the sample's water map
         - iterative: bool: -- whether to vary the time difference between the sample and label to create all sample-label pairs
                                such that the label is no farther than timestep number of months from the label
                                e.x: if timestep = 3, the dataset will contain sample-label pairs: (1, 4), (1, 3), (1, 2) where the timestep gap between sample and label is no more than timestep
         """
        
        self.data_path = data_path
        self.timestep = timestep
        self.iterative = iterative
        self.transform = transform
        self.target_transform = target_transform
        self.num_files = len(os.listdir(self.data_path))
        self.min_year, self.min_month = min([int(filename[9:13]) for filename in os.listdir(self.data_path)]), min([int(filename[14:-4]) for filename in os.listdir(self.data_path)])
        self.dtype=dtype
        self.device=device

    def __len__(self):
        # calculate the length differently based on whether we iteratively choose a label or statically
        if self.iterative:
            return int(self.num_files * self.timestep - (self.timestep + 1) * self.timestep/2)
        else:
            return self.num_files - self.timestep
        
    def __getitem__(self, idx):
        # iterative label selection
        if self.iterative:

            # if the index belongs to a sample on the edge of the dataset, treat the indexing differently so that we don't select an out of bounds label
            if idx >= (self.num_files - self.timestep) * self.timestep:
                data_month_from_start = self.num_files - self.timestep
                excess_idx = idx - (self.num_files - self.timestep) * self.timestep
                for width in range(self.timestep - 1, 0, -1):
                    if excess_idx - width >= 0:
                        excess_idx -= width
                        data_month_from_start += 1
                        continue
                    label_month_from_start = data_month_from_start + excess_idx + 1
                    break

            # otherwise compute the sample and label by a simple operation on the index 
            else:
                data_month_from_start = idx // self.timestep
                label_month_from_start = data_month_from_start + 1 + idx % self.timestep

            # compute the data and label's month and year, and also the gap between them in months
            data_year = self.min_year + data_month_from_start // 12
            data_month = 1 + data_month_from_start % 12

            label_year = self.min_year + label_month_from_start // 12
            label_month = 1 + label_month_from_start % 12

            label_month_gap = label_month_from_start  - data_month_from_start

        # static label selection
        else:
            # compute the data and label's month and year, and also the gap between them in months
            label_month_gap = self.timestep
            data_year = self.min_year + idx // 12
            data_month = 1 + idx % 12

            label_year = self.min_year + (idx + self.timestep) // 12
            label_month = 1 + (idx + self.timestep) % 12


        # read from the appropriate files
        data = torch.from_numpy(np.load(os.path.join(self.data_path, f'amz_near_{data_year}_{data_month}.npy'))).to(dtype=self.dtype, device=self.device)
        label = torch.from_numpy(np.load(os.path.join(self.data_path, f'amz_near_{label_year}_{label_month}.npy'))).to(dtype=self.dtype, device=self.device)


        # apply appropriate transforms
        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            label = self.target_transform(label)

        label_month_gap = torch.tensor(label_month_gap, dtype=self.dtype).unsqueeze(0)

        return data.to(self.device), label.to(self.device), label_month_gap.to(self.device)



