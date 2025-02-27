import os
import torch
import numpy as np
import pandas as pd
from glob import glob
import re
from monai.transforms import (
    Compose, 
    RandFlipd, 
    RandRotated, 
    ScaleIntensityd,
    ToTensord,
    LoadImaged,
    EnsureChannelFirstd,
    Resized,
    Spacingd
)
import monai.transforms as transforms
import monai

from torch.utils.data import DataLoader

# Set random seed for reproducibility
monai.utils.set_determinism(seed=42)

class HipFusionDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, label_path, augment=False, sheet_name=None, n=None):
        """
        Dataset for hip MRI analysis with fusion of image and metadata.
        
        Args:
            img_path: Path to directory containing .nii or .nii.gz files
            label_path: Path to Excel file containing labels and metadata
            augment: Whether to apply data augmentation
            n: Optional limit on the number of samples
        """
        self.img_path = img_path
        self.label_path = label_path
        self.augment = augment
        self.n = n
        self.sheet_name = sheet_name
        self.meta_features = 4
        
        # Load Excel file with labels
        self.labels_df = pd.read_excel(self.label_path, sheet_name=self.sheet_name)
        
        # Storage for matched files and their corresponding information
        self.file_paths = []
        self.metadata_list = []
        self.labels_list = []
        
        # Find all NIfTI files in the directory
        all_nii_files = glob(os.path.join(self.img_path, "*.nii.gz"))
        if not all_nii_files:
            all_nii_files = glob(os.path.join(self.img_path, "*.nii"))
        
        # Match NIfTI files with corresponding entries in the Excel file
        for nii_file in all_nii_files:
            # Extract study name and side from filename
            filename = os.path.basename(nii_file)
            match = re.search(r'img(.*?)_annotation-(l|r)', filename)
            
            if match:
                study_name = match.group(1)
                side = match.group(2)
                
                # Find corresponding row in Excel file
                matching_row = self.labels_df[(self.labels_df['Study number'] == study_name) & 
                                            (self.labels_df['Side'] == side)]
                
                if not matching_row.empty:
                    self.file_paths.append(nii_file)
                    
                    # Extract metadata: Age, Gender, Treatment, Luxation
                    # Convert Gender to binary representation (m=0, f=1)
                    gender_value = matching_row['Gender']
                    
                    metadata = [
                        matching_row['Age'].values[0],
                        gender_value,
                        matching_row['Treatment'].values[0],
                        matching_row['Luxation'].values[0]
                    ]
                    
                    self.metadata_list.append(metadata)
                    
                    # Extract THP label (0 or 1)
                    self.labels_list.append(matching_row['THP'].values[0])
        
        # Define transforms
        # Base transforms applied to all samples
        base_transforms = [
            LoadImaged(keys=["image"]),               # Load NIfTI file
            EnsureChannelFirstd(keys=["image"]),      # Ensure channel is first dimension
            Resized(keys=["image"], spatial_size=(64, 64, 64)),  # Resize to 64x64x64 voxels
            ScaleIntensityd(keys=["image"]),          # Normalize intensity values
            ToTensord(keys=["image"])                 # Convert to PyTorch tensor
        ]
        
        # Additional augmentation transforms
        augment_transforms = [
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=[0, 1, 2]),  # Random flip along any axis
            RandRotated(keys=["image"], range_x=15, range_y=15, range_z=15, 
                       prob=0.7, mode="bilinear")     # Random rotation in 3D space
        ]
        
        # Apply different transform pipelines based on augmentation flag
        if self.augment:
            self.transform = Compose(base_transforms + augment_transforms)
        else:
            self.transform = Compose(base_transforms)
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        if self.n is not None:
            return min(self.n, len(self.file_paths))
        else:
            return len(self.file_paths)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing:
                - image: 3D tensor of shape [C, 64, 64, 64]
                - metadata: tensor of metadata features [Age, Gender, Treatment, Luxation]
                - label: tensor containing THP class (0 or 1)
        """
        # Load NIfTI file
        file_path = self.file_paths[idx]
        metadata = self.metadata_list[idx]
        label = self.labels_list[idx]
        
        # Apply transforms
        transformed = self.transform({"image": file_path})
        x = transformed["image"]
        
        # Convert metadata and label to tensors
        meta = torch.tensor(metadata, dtype=torch.float)
        y = torch.tensor(label, dtype=torch.float)
        
        return {"image": x, "metadata": meta, "label": y}

def main():
    """
    Test function to verify the HipFusionDataset class functionality
    """
    # Set paths
    img_path = "/data/hip/2year_training"
    label_path = "/data/hip/Classification_THP_1301_modified.xlsx"
    
    # Create dataset instance
    dataset = HipFusionDataset(img_path=img_path, label_path=label_path, augment=False, sheet_name="2year model")
    
    # Print total number of samples found
    print(f"Total samples found: {len(dataset)}")
    
    # Create a DataLoader to iterate through samples
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Print information for each sample
    for i, batch in enumerate(dataloader):
        image = batch["image"]
        metadata = batch["metadata"]
        label = batch["label"]
        
        # Print details
        print(f"\nSample {i+1}:")
        print(f"  File: {dataset.file_paths[i]}")
        print(f"  Image shape: {image.shape}")
        print(f"  Metadata: {metadata.numpy().flatten()}")
        print(f"  Label: {label.item()}")
        
        # Print metadata interpretation
        age, gender, treatment, luxation = metadata.numpy().flatten()
        gender_text = "Female" if gender == 1 else "Male"
        print(f"  Metadata interpretation:")
        print(f"    Age: {age}")
        print(f"    Gender: {gender_text} ({gender})")
        print(f"    Treatment: {treatment}")
        print(f"    Luxation: {luxation}")
        
        # Limit to first 5 samples for brevity
        if i >= 4:
            remaining = len(dataset) - 5
            print(f"\n... and {remaining} more samples")
            break

    # Calculate class distribution
    label_counts = {0: 0, 1: 0}
    for i in range(len(dataset)):
        label_val = dataset.labels_list[i]
        label_counts[label_val] += 1
    
    print("\nClass distribution:")
    print(f"  Class 0: {label_counts[0]} samples ({100 * label_counts[0]/len(dataset):.1f}%)")
    print(f"  Class 1: {label_counts[1]} samples ({100 * label_counts[1]/len(dataset):.1f}%)")


if __name__ == "__main__":
    main()
