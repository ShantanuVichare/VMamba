import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split, KFold


# Dataset for 3D MRI images
class TumorMRIDataset(Dataset):
    def __init__(self, root_dir, limit=None):
        self.root_dir = root_dir
        self.modalities = ['t1ce', 't1', 't2', 'flair', 'seg']
        self.samples = self._load_samples(root_dir, limit)

    def _load_samples(self, root_dir, limit=None):
        samples = []
        for label in ['HGG', 'LGG']:
            label_specific_sample_count = 0
            folder_path = os.path.join(root_dir, label)
            for patient_folder in os.listdir(folder_path):
                file_paths = [None] * len(self.modalities)
                for i, modality in enumerate(self.modalities):
                    file_paths[i] = os.path.join(folder_path, patient_folder, f'{patient_folder}_{modality}.nii')
                samples.append((file_paths, 0 if label == 'HGG' else 1))
                label_specific_sample_count += len(file_paths)
                if limit is not None and label_specific_sample_count >= limit:
                    break
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_paths, label = self.samples[idx]
        tensor_list = []
        for file_path in file_paths:
            img = nib.load(file_path).get_fdata()
            img = self._pad_or_crop(img)
            tensor_list.append(torch.tensor(img, dtype=torch.float32).permute(2,0,1))
        return torch.stack(tensor_list), torch.tensor(label, dtype=torch.long)

    def _pad_or_crop(self, img):
        target_shape = img.shape
        pad_size = [(max(0, target - img_dim)) for target, img_dim in zip(target_shape, img.shape)]
        pad_widths = [(p // 2, p - p // 2) for p in pad_size]
        img_padded = np.pad(img, pad_widths, mode='constant', constant_values=0)
        return img_padded[:target_shape[0], :target_shape[1], :target_shape[2]]


# Split the dataset into train and test sets by class
def split_dataset_by_class(dataset, train_ratio=0.8):
    HGG_samples = [sample for sample in dataset.samples if sample[1] == 0]
    LGG_samples = [sample for sample in dataset.samples if sample[1] == 1]
    

    # Split each class
    HGG_train, HGG_test = train_test_split(HGG_samples, train_size=train_ratio, shuffle=True)
    LGG_train, LGG_test = train_test_split(LGG_samples, train_size=train_ratio, shuffle=True)

    # Combine the train and test samples
    train_samples = HGG_train + LGG_train
    test_samples = HGG_test + LGG_test
    distribution_info = {'HGG_train': len(HGG_train), 'HGG_test': len(HGG_test), 'LGG_train': len(LGG_train), 'LGG_test': len(LGG_test)}

    return train_samples, test_samples, distribution_info
