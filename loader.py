import math
from PIL import Image

import torch
import torch.utils.data as data
import torch.distributed as dist
from torch.utils.data import Sampler


class CategoryDataset(data.Dataset):
    def __init__(self, data_path, transforms):
        data_file = open(data_path)

        self.transforms = transforms

        self.images = []
        self.labels = []
        try:
            text_lines = data_file.readlines()
            for i in text_lines:
                i = i.strip()
                self.images.append(i.split(' ')[0])
                self.labels.append(int(i.split(' ')[1]))
        finally:
            data_file.close()

    def __getitem__(self, ind):
        image = Image.open(self.images[ind])
        label = self.labels[ind]

        image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.images)
