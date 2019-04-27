import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import random

from torch.utils import data

from ptsemseg.utils import recursive_glob, read_file


class ADE20KFewShotLoader(data.Dataset):
    def __init__(
        self,
        root,
        is_transform=False,
        img_size=512,
        augmentations=None,
        img_norm=True,
        test_mode=False,
    ):
        self.root = root
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.n_classes = 6
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])

        if not self.test_mode:
            image_list = recursive_glob(rootdir=self.root + "images/training/", suffix=".jpg")
            annotation_list = recursive_glob(rootdir=self.root + "annotations/training/", suffix='.png')
            presentation_list = read_file(rootdir=self.root + "presentations/training/", filename='presentations.txt', split=',')
            classes_list = read_file(rootdir=self.root + "presentations/training/", filename="presentation_classes.txt", split=',')
        else:
            image_list = recursive_glob(rootdir=self.root + "images/validation/", suffix='.jpg')
            annotation_list = recursive_glob(rootdir=self.root + "annotations/validation/", suffix='.png')
            presentation_list = read_file(rootdir=self.root + "presentations/validation/", filename='presentations.txt', split=',')
            classes_list = read_file(rootdir=self.root + "presentations/validation/", filename='presentation_classes.txt', split=',')

        self.images = image_list
        self.annotations = annotation_list
        self.presentations = presentation_list
        self.pre_classes = classes_list

    def __len__(self):
        return len(self.presentations[0])

    def __getitem__(self, index):
        index += 1
        index %= self.__len__()
        if index == 1:
            # if all images in one presentation has been visited once
            # randomly select a new presentation from the presentation list
            idx = random.randint(0, len(self.presentations) - 1)
            self.presentation = self.presentations[idx]
            # get the common classes of the selected presentation
            self.classes = self.pre_classes[idx]

        img_path = [image for image in self.images if self.presentation[index] in image]
        ann_path = [ann for ann in self.annotations if self.presentation[index] in ann]

        img = m.imread(img_path[0])
        img = np.array(img, dtype=np.uint8)

        ann = m.imread(ann_path[0])
        ann = np.array(ann, dtype=np.int32)
        ann = self.zero_annotation(ann)

        if self.augmentations is not None:
           img, ann = self.augmentations(img, ann)

        if self.is_transform:
            img, ann = self.transform(img, ann)

        return img, ann

    def zero_annotation(self, annotation):
        annotation[np.isin(annotation, self.classes, invert=True)] = 0
        return annotation

    def transform(self, img, lbl):
        img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)
        assert np.all(classes == np.unique(lbl))

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl


if __name__ == "__main__":
    local_path = "./loader_test/"
    dst = ADE20KFewShotLoader(local_path, is_transform=True)
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