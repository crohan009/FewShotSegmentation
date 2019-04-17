import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

from torch.utils import data
import os
import random


class ADE20KLoader_Zhou(data.Dataset):
    def __init__(
        self,
        root,
        split="training",
        is_transform=False,
        img_size=512,
        augmentations=None,
        img_norm=True,
        test_mode=False,
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.n_classes = 151
        self.img_size = img_size[0]
        self.img_transform = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.files = collections.defaultdict(list)

        if not self.test_mode:
            # for split in ["training", "validation"]:
            file_list = os.listdir(self.root + "images/" + self.split + "/")
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = os.path.join(self.root + "images/" + self.split + "/", self.files[self.split][index].rstrip())
        lbl_path = os.path.join(self.root + "annotations/" + self.split + "/",
                                self.files[self.split][index].replace('.jpg', '.png').rstrip())

        try:
            img = imread(img_path, mode='RGB')
            seg = imread(lbl_path)
            assert(img.ndim == 3)
            assert(seg.ndim == 2)
            assert(img.shape[0] == seg.shape[0])
            assert(img.shape[1] == seg.shape[1])

            # random scale, crop, flip
            if self.img_size > 0:
                img, seg = self._scale_and_crop(img, seg,
                                                self.img_size, self.test_mode)
                if random.choice([-1, 1]) > 0:
                    img, seg = self._flip(img, seg)

            # image to float
            img = img.astype(np.float32) / 255.
            img = img.transpose((2, 0, 1))

            # label to int from -1 to 149
            seg = seg.astype(np.int)

            # to torch tensor
            image = torch.from_numpy(img)
            segmentation = torch.from_numpy(seg)
        except Exception as e:
            print('Failed loading image/segmentation [{}]: {}'
                  .format(img_path, e))
            # dummy data
            image = torch.zeros(3, self.img_size, self.img_size)
            segmentation = -1 * torch.ones(self.img_size, self.img_size).long()
            return image, segmentation

        image = self.img_transform(image)

        return image, segmentation

    def _scale_and_crop(self, img, seg, cropSize, test_mode):
        h, w = img.shape[0], img.shape[1]

        if not test_mode:
            # random scale
            scale = random.random() + 0.5  # 0.5-1.5
            scale = max(scale, 1. * cropSize / (min(h, w) - 1))
        else:
            # scale to crop size
            scale = 1. * cropSize / (min(h, w) - 1)

        img_scale = imresize(img, scale, interp='bilinear')
        seg_scale = imresize(seg, scale, interp='nearest')

        h_s, w_s = img_scale.shape[0], img_scale.shape[1]
        if not test_mode:
            # random crop
            x1 = random.randint(0, w_s - cropSize)
            y1 = random.randint(0, h_s - cropSize)
        else:
            # center crop
            x1 = (w_s - cropSize) // 2
            y1 = (h_s - cropSize) // 2

        img_crop = img_scale[y1: y1 + cropSize, x1: x1 + cropSize, :]
        seg_crop = seg_scale[y1: y1 + cropSize, x1: x1 + cropSize]
        return img_crop, seg_crop

    def _flip(self, img, seg):
        img_flip = img[:, ::-1, :]
        seg_flip = seg[:, ::-1]
        return img_flip, seg_flip


if __name__ == "__main__":
    local_path = "/Users/meet/data/ADE20K_2016_07_26/"
    dst = ADE20KLoader_Zhou(local_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1]
            plt.imshow(img)
            plt.show()
            for j in range(4):
                plt.imshow(dst.decode_segmap(labels.numpy()[j]))
                plt.show()
