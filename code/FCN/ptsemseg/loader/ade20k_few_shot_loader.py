import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import random

from torch.utils import data

from ptsemseg.utils import recursive_glob, read_file
from ptsemseg.augmentations import Compose, RandomRotate


class ADE20KFewShotLoader(data.Dataset):
    def __init__(
        self,
        data_root="",
        presentation_root="",
        is_transform=False,
        img_size=512,
        augmentations=None,
        aug_k=4,
        img_norm=True,
        test_mode=False,
    ):
        self.data_root = data_root
        self.presentation_root = presentation_root
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.aug_k = aug_k
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.n_classes = 6
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])

        if not self.test_mode:
            image_list = recursive_glob(rootdir=self.data_root + "images/training/", suffix=".jpg")
            annotation_list = recursive_glob(rootdir=self.data_root + "annotations/training/", suffix='.png')
            presentation_list = read_file(rootdir=self.presentation_root, filename='train_presentations.txt', split=',')
            classes_list = read_file(rootdir=self.presentation_root, filename="train_class_list.txt", split=',')
        else:
            image_list = recursive_glob(rootdir=self.data_root + "images/validation/", suffix='.jpg')
            annotation_list = recursive_glob(rootdir=self.data_root + "annotations/validation/", suffix='.png')
            presentation_list = read_file(rootdir=self.presentation_root, filename='val_presentations.txt', split=',')
            classes_list = read_file(rootdir=self.presentation_root, filename='val_class_list.txt', split=',')

        self.images = image_list
        self.annotations = annotation_list
        self.presentations = presentation_list
        self.pre_classes = classes_list

        self.presentation = []
        self.classes = []
        self.image_data = []
        self.label_data = []

    def __len__(self):
        if self.augmentations is None:
            return len(self.presentations[0])
        else:
            return (len(self.presentations[0]) - 1) * self.aug_k + 1

    def __getitem__(self, index):
        index += 1
        index %= self.__len__()

        return torch.from_numpy(self.image_data[index]).float(), torch.from_numpy(self.label_data[index]).long()
    
    def random_select(self):
        idx = random.randint(0, len(self.presentations) - 1)
        self.presentation = self.presentations[idx]
        self.classes = [int(c) for c in self.pre_classes[idx]]

        for index in range(len(self.presentation)):
            img_path = [image for image in self.images if self.presentation[index] in image]
            ann_path = [ann for ann in self.annotations if self.presentation[index] in ann]

            img = m.imread(img_path[0])
            if img.ndim == 2:
                list = []
                for i in range(3):
                    list.append(img)
                img = list
            img = np.array(img, dtype=np.uint8)

            ann = m.imread(ann_path[0])
            ann = np.array(ann, dtype=np.int32)
            ann = self.modify_annotation(ann)

            # transformation
            if self.is_transform:
                img, ann = self.transform(img, ann)

            # augmentation
            if self.augmentations is not None and index != 0:
                for k in range(self.aug_k):
                    new_img, new_ann = self.augmentations(img, ann)
                    # NHWC -> NCHW
                    new_img = new_img.transpose(2, 0, 1)
                    self.image_data.append(new_img)
                    self.label_data.append(new_ann)

            # NHWC -> NCHW
            img = img.transpose(2, 0, 1)
            self.image_data.append(img)
            self.label_data.append(ann)

        self.image_data = np.array(self.image_data)
        self.label_data = np.array(self.label_data)

        shuffle_slice = [0]
        shuffle_slice.extend(np.random.permutation(range(1, self.image_data.shape[0])))
        self.image_data = self.image_data[shuffle_slice]
        self.label_data = self.label_data[shuffle_slice]

    def modify_annotation(self, annotation):
        annotation[np.isin(annotation, self.classes, invert=True)] = 0
        for c in self.classes:
            annotation[annotation==c] = self.classes.index(c) + 1
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

        # classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype('uint8')
        # assert np.all(classes == np.unique(lbl))

        return img, lbl


if __name__ == "__main__":
    data_path = "./loader_test/"
    presentation_path = "./loader_test/presentations/training"
    dst = ADE20KFewShotLoader(data_root=data_path, presentation_root=presentation_path, is_transform=True)
    trainloader = data.DataLoader(dst, batch_size=4)
    trainloader.dataset.random_select()
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
