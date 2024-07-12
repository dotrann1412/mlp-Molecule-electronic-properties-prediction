from torch.utils.data import Dataset 
import torch

class MoleculeDataset(Dataset):
    def __init__(self, data: list, transforms = None, labels: list = None):
        self.data = data
        self.labels = labels
        self.transforms = transforms

    def is_training(self):
        return self.labels is not None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        sample = self.data[idx]

        if self.transforms:
            sample = self.transforms(sample) # np.array

        sample = torch.tensor(sample, dtype=torch.float32)

        if self.is_training():
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.float32) 
            return sample, label

        return sample, None