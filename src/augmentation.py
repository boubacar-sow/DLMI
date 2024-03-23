from pathlib import Path
import torch
from torchvision.transforms import RandomHorizontalFlip
from torch.utils.data import ConcatDataset

from dataloader import SegmentationDataset

def augmentation(dataset):
    transform = RandomHorizontalFlip(p=1)  

    augmented_data = []
    for image, label in dataset:
        image = image.clone().detach()
        label = label.clone().detach() if label is not None else None
        augmented_image = transform(image)
        augmented_label = transform(label) if label is not None else None

        augmented_data.append((augmented_image, augmented_label))
    
    images, labels = zip(*augmented_data)
    images = torch.stack(images)
    labels = torch.stack(labels) if labels[0] is not None else None
    augmented_dataset = torch.utils.data.TensorDataset(images, labels)
    combined_dataset = ConcatDataset([dataset, augmented_dataset])

    return combined_dataset


if __name__ == "__main__":
    # Test the augmentation
    train_image_path = Path('./data/Train/images')
    train_label_path = Path('./data/Train/labels')
    train_dataset = SegmentationDataset(train_image_path, train_label_path)
    
    augmented_train_dataset = augmentation(train_dataset)
    
    print(f"Original train dataset size: {len(train_dataset)}")
    print(f"Augmented train dataset size: {len(augmented_train_dataset)}")