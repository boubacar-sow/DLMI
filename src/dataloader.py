import torch
import cv2
import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class LymphBags(Dataset):
    def __init__(self, bags_dir, df, indices, mode='train', transforms=None):
        assert mode in ['train', 'test'], "mode must belong to ['train', 'test']"
        self.transforms = transforms
        self.mode = mode
        self.df = df
        self.dir = bags_dir
        self.bags = list(filter(lambda x: x[0] == 'P', os.listdir(bags_dir)))
        self.bags = [self.bags[i] for i in indices]

    def __len__(self):
        return len(self.bags)

    def load_images(self, bags):
        images = []
        for bag in os.listdir(bags):
            img = cv2.imread(os.path.join(bags, bag))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            if self.transforms:
                img = self.transforms(img)
            images.append(img)

        images = torch.stack(images)

        return images

    def __getitem__(self, index):
        bags = os.path.join(self.dir, self.bags[index])
        idx_ = self.df[self.df['ID'] == self.bags[index]].index[0]
        images = self.load_images(bags)
        gender = torch.as_tensor([self.df.iloc[idx_, 2]])
        count = torch.as_tensor([self.df.iloc[idx_, 4]])
        age = torch.as_tensor([self.df.iloc[idx_, -1]])

        if self.mode == 'train':
            label = self.df.iloc[idx_, 1]
            return images, gender, count, age, label
        else:
            return images, gender, count, age, idx_
